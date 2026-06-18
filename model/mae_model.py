

import torch
import torch.nn as nn

from tokenizer import GaussianTokenizer, gather_points


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(self, x):
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x



def random_masking(tokens, mask_ratio):
    """
    Random masking 함수.
    
    Args:
        tokens: (B, G, D)
        mask_ratio: 마스킹 비율 (예: 0.6 = 60%)
    
    Returns:
        visible_tokens: (B, G_visible, D)
        mask: (B, G) — True가 masked, False가 visible
        ids_restore: (B, G) — 원래 순서 복원용 인덱스
        ids_keep: (B, G_visible) — visible 토큰의 원본 인덱스
    """
    B, G, D = tokens.shape
    n_keep = int(G * (1 - mask_ratio))
    
    # 각 batch마다 랜덤 노이즈 → 작은 값 우선 = visible
    noise = torch.rand(B, G, device=tokens.device)
    
    # 노이즈로 정렬 → 작은 게 visible, 큰 게 masked
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Visible 인덱스
    ids_keep = ids_shuffle[:, :n_keep]
    
    # Visible 토큰 추출
    visible_tokens = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    # mask: True가 masked
    mask = torch.ones(B, G, device=tokens.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore).bool()
    
    return visible_tokens, mask, ids_restore, ids_keep



class GaussianMAE(nn.Module):
    """
    3DGS Masked Autoencoder.
    
    Encoder는 visible token만, Decoder는 visible + mask token 모두 처리.
    """
    def __init__(
        self,
        # Tokenizer
        num_groups=1024,
        group_size=32,
        input_dim=15,
        embed_dim=384,
        # Encoder
        encoder_depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        # Decoder
        decoder_depth=2,
        # MAE
        mask_ratio=0.60,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        
        # Tokenizer
        self.tokenizer = GaussianTokenizer(
            num_groups=num_groups,
            group_size=group_size,
            input_dim=input_dim,
            embed_dim=embed_dim,
        )
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Mask token (학습 가능 placeholder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Decoder position embedding (centers xyz로부터)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        
        # Reconstruction head: token → patch (group_size × input_dim)
        self.recon_head = nn.Linear(embed_dim, group_size * input_dim)
    
    def forward(self, params, return_all=False):
        """
        Args:
            params: (B, N, 14) Gaussian 파라미터
            return_all: True면 중간 결과도 반환 (eval 시 유용)
        
        Returns:
            recon_patches: (B, G_masked, group_size, input_dim)
                          마스킹된 patch들의 복원 결과
            target_patches: (B, G_masked, group_size, input_dim)
                           원본 patch (loss 계산용)
            mask: (B, G) — True가 masked
            centers: (B, G, 14)
            patch_idx: (B, G, k)
            (return_all=True 시) recon_full: (B, G, group_size, input_dim) 전체 복원
        """
        B = params.shape[0]
        
        # 1. Tokenize
        tokens, centers, patch_idx = self.tokenizer(params)
        # tokens: (B, G, embed_dim), centers: (B, G, 14), patch_idx: (B, G, k)
        
        # 원본 patches 추출 (loss 계산용)
        # gather_points로 (B, G, k, 14) 형태
        target_patches = gather_points(params, patch_idx)  # (B, G, k, 14)
        # patch 내부를 center 기준 상대 좌표로
        target_patches_norm = target_patches.clone()
        target_patches_norm[:, :, :, :3] = (
            target_patches[:, :, :, :3] - centers[:, :, :3].unsqueeze(2)
        )
        
        # 2. Random masking
        visible_tokens, mask, ids_restore, ids_keep = random_masking(tokens, self.mask_ratio)
        # visible_tokens: (B, G_visible, embed_dim)
        # mask: (B, G) True가 masked
        
        # 3. Encoder (visible만)
        x = visible_tokens
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        # x: (B, G_visible, embed_dim)
        
        # 4. Decoder 입력 준비: visible + mask token을 원래 순서로 배치
        G = self.num_groups
        n_keep = ids_keep.shape[1]
        n_mask = G - n_keep
        
        # Mask token 배치
        mask_tokens = self.mask_token.expand(B, n_mask, -1)  # (B, n_mask, embed_dim)
        
        # Visible + mask 합치기 (이때는 visible이 앞쪽)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, G, embed_dim)
        
        # 원래 순서로 복원
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )  # (B, G, embed_dim)
        
        # Position embedding 추가
        decoder_pos = self.decoder_pos_embed(centers[:, :, :3])  # (B, G, embed_dim)
        x_full = x_full + decoder_pos
        
        # 5. Decoder
        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)
        # x_full: (B, G, embed_dim)
        
        # 6. Reconstruction head
        recon_full = self.recon_head(x_full)  # (B, G, group_size * input_dim)
        recon_full = recon_full.reshape(B, G, self.group_size, self.input_dim)
        
        # Masked 부분만 추출 (loss는 masked만 계산)
        recon_patches = recon_full[mask].reshape(B, n_mask, self.group_size, self.input_dim)
        target_patches_masked = target_patches_norm[mask].reshape(B, n_mask, self.group_size, self.input_dim)
        
        if return_all:
            return {
                'recon_patches': recon_patches,
                'target_patches': target_patches_masked,
                'recon_full': recon_full,             # 전체 복원
                'target_full': target_patches_norm,   # 전체 원본
                'mask': mask,
                'centers': centers,
                'patch_idx': patch_idx,
            }
        
        return recon_patches, target_patches_masked, mask, centers, patch_idx



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 모델
    model = GaussianMAE(
        num_groups=1024,
        group_size=32,
        input_dim=15,
        embed_dim=384,
        encoder_depth=6,
        decoder_depth=2,
        num_heads=6,
        mask_ratio=0.60,
    ).to(device)
    
    print(f"\n파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 가짜 데이터
    B, N = 2, 20000
    params = torch.randn(B, N, 15).to(device)
    print(f"\n입력: {params.shape}")
    
    # Forward
    recon, target, mask, centers, patch_idx = model(params)
    print(f"\n[Output (학습 모드)]")
    print(f"  recon_patches: {recon.shape}")
    print(f"  target_patches: {target.shape}")
    print(f"  mask: {mask.shape}, masked 비율: {mask.float().mean().item():.2%}")
    print(f"  centers: {centers.shape}")
    print(f"  patch_idx: {patch_idx.shape}")
    
    # Loss 계산 테스트
    loss = ((recon - target) ** 2).mean()
    print(f"\n[Loss 테스트]")
    print(f"  L2 loss: {loss.item():.6f}")
    
    # 역전파 테스트
    loss.backward()
    print(f" Backward 성공")
    
    # Eval 모드 (return_all=True)
    print(f"\n[Eval 모드]")
    with torch.no_grad():
        out = model(params, return_all=True)
    print(f"  recon_full: {out['recon_full'].shape}")
    print(f"  target_full: {out['target_full'].shape}")
    
    print("\n MAE 모델 동작 확인 완료!")