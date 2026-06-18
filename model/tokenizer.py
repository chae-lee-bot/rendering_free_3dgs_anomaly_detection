

import torch
import torch.nn as nn

@torch.no_grad()
def farthest_point_sample(xyz, n_samples):

    device = xyz.device
    B, N, _ = xyz.shape
    
    # 결과 인덱스 저장
    idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    
    # 각 점까지의 거리 (초기엔 무한대)
    distance = torch.full((B, N), 1e10, device=device)
    
    # 첫 점은 랜덤 선택
    farthest = torch.randint(0, N, (B,), device=device)
    
    batch_indices = torch.arange(B, device=device)
    
    for i in range(n_samples):
        idx[:, i] = farthest
        # 현재 farthest 점의 좌표
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        # 모든 점과의 거리 제곱
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        # 각 점의 "가장 가까운 center까지의 거리" 업데이트
        distance = torch.minimum(distance, dist)
        # 가장 먼 점을 다음 center로
        farthest = torch.argmax(distance, dim=-1)
    
    return idx



@torch.no_grad()
def knn_group(xyz, centers_xyz, k):
    # pairwise 거리: (B, G, N)
    dists = torch.cdist(centers_xyz, xyz)  # L2 distance
    # 가장 가까운 k개
    _, idx = torch.topk(dists, k, dim=-1, largest=False)  # (B, G, k)
    return idx


def gather_points(params, idx):
    B, N, D = params.shape
    
    if idx.dim() == 3:
        # (B, G, k) → (B, G, k, D)
        G, k = idx.shape[1], idx.shape[2]
        idx_flat = idx.reshape(B, -1)  # (B, G*k)
        # batch별 gather
        idx_expand = idx_flat.unsqueeze(-1).expand(-1, -1, D)  # (B, G*k, D)
        gathered = torch.gather(params, 1, idx_expand)  # (B, G*k, D)
        return gathered.reshape(B, G, k, D)
    else:
        # (B, G) → (B, G, D)
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(params, 1, idx_expand)


class MiniPointNet(nn.Module):
    """
    구조: per-point MLP → max-pool over patch
    """
    def __init__(self, input_dim=15, embed_dim=384):
        super().__init__()
        # Per-point feature extractor
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )
        # Global feature aggregator
        self.mlp2 = nn.Sequential(
            nn.Linear(256 + 256, 512),  # local + max-pool concatenation
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, G, k, D) patch 데이터
        Returns:
            tokens: (B, G, embed_dim)
        """
        B, G, k, D = x.shape
        x = x.reshape(B * G, k, D)
        
        # Per-point features
        feat = self.mlp1(x)  # (B*G, k, 256)
        
        # Max-pool (global feature)
        global_feat = feat.max(dim=1, keepdim=True)[0]  # (B*G, 1, 256)
        global_feat = global_feat.expand(-1, k, -1)  # (B*G, k, 256)
        
        # Concat local + global
        feat = torch.cat([feat, global_feat], dim=-1)  # (B*G, k, 512)
        
        # Final MLP + max-pool
        feat = self.mlp2(feat)  # (B*G, k, embed_dim)
        tokens = feat.max(dim=1)[0]  # (B*G, embed_dim)
        
        return tokens.reshape(B, G, -1)



class GaussianTokenizer(nn.Module):

    def __init__(self, num_groups=1024, group_size=32, input_dim=15, embed_dim=384):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.embed_dim = embed_dim
        
        self.embedding = MiniPointNet(input_dim=input_dim, embed_dim=embed_dim)
        
        # Position encoding (xyz 기반)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
    
    def forward(self, params):
        B, N, D = params.shape
        xyz = params[:, :, :3]  # (B, N, 3)
        
        # 1. FPS로 center 인덱스 선택
        center_idx = farthest_point_sample(xyz, self.num_groups)  # (B, G)
        centers = gather_points(params, center_idx)  # (B, G, 14)
        centers_xyz = centers[:, :, :3]  # (B, G, 3)
        
        # 2. KNN grouping
        patch_idx = knn_group(xyz, centers_xyz, self.group_size)  # (B, G, k)
        patches = gather_points(params, patch_idx)  # (B, G, k, 14)
        
        # 3. Patch 내부를 center 기준 상대 좌표로 변환 (translation invariance)
        patches_norm = patches.clone()
        patches_norm[:, :, :, :3] = patches[:, :, :, :3] - centers_xyz.unsqueeze(2)
        
        # 4. Embedding
        tokens = self.embedding(patches_norm)  # (B, G, embed_dim)
        
        # 5. Position embedding (center xyz 기반)
        pos = self.pos_embed(centers_xyz)  # (B, G, embed_dim)
        tokens = tokens + pos
        
        return tokens, centers, patch_idx


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 가짜 데이터
    B, N, D = 2, 20000, 15
    params = torch.randn(B, N, D).to(device)
    
    # Tokenizer
    tokenizer = GaussianTokenizer(
        num_groups=1024,
        group_size=32,
        input_dim=15,
        embed_dim=384,
    ).to(device)
    
    print(f"\n파라미터 수: {sum(p.numel() for p in tokenizer.parameters()):,}")
    
    print(f"\n입력: {params.shape}")
    tokens, centers, patch_idx = tokenizer(params)
    print(f"Tokens: {tokens.shape}")        # (B, G, embed_dim)
    print(f"Centers: {centers.shape}")       # (B, G, 14)
    print(f"Patch idx: {patch_idx.shape}")   # (B, G, k)
    