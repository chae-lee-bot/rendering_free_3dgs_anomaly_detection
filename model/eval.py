"""
학습된 MAE로 anomaly detection 평가.
- Overall AUROC
- 타입별 AUROC (burrs/stains/missing)
- 클래스별 AUROC
- 클래스별 + 타입별 AUROC (상세 분석)
- 결과를 npz/json으로 저장하도록 -> 설정 완료
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KDTree
from plyfile import PlyData
import glob

from configs import Config
from dataset import AnomalyGaussianDataset, NormalGaussianDataset, ATTR_SLICES, GaussianNormalizer
from mae_model import GaussianMAE



def _load_normal_xyz(data_root, cls):
    """클래스의 정상 PLY 1개에서 xyz만 로드."""
    cands = sorted(glob.glob(os.path.join(data_root, cls, 'normal', 'point_cloud_*.ply')))
    assert len(cands) > 0, f"정상 PLY 없음: {cls}"
    ply = PlyData.read(cands[0])
    v = ply['vertex']
    return np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=1).astype(np.float32)


def precompute_class_radii(classes, data_root, k_radius=2.5):
    """
    각 클래스마다 정상 PLY의 NN 평균 거리 × k_radius 를 미리 계산.
    Returns: {cls: r_thresh(float)}
    """
    radii = {}
    print(f"\n[정상 NN 평균 × {k_radius} 계산 중...]")
    for cls in classes:
        xyz = _load_normal_xyz(data_root, cls)
        tree = KDTree(xyz)
        # k=2: 자기 자신(거리 0) + 가장 가까운 이웃 1개
        dist, _ = tree.query(xyz, k=2)
        nn_dist = dist[:, 1]  # 자기 자신 제외
        nn_mean = float(nn_dist.mean())
        radii[cls] = nn_mean * k_radius
        print(f"  {cls:15s}: N={len(xyz):6d}, NN_mean={nn_mean:.5f}, r_thresh={radii[cls]:.5f}")
    return radii


def compute_missing_gt(xyz_orig, removed_xyz, r_thresh):
    """
    Missing GT 재계산: 남아있는 가우시안 중 removed_xyz 점들로부터 r_thresh 이내에 있는 것을 1로.
    
    xyz_orig:    (N, 3)  정규화 전 원본 xyz
    removed_xyz: (M, 3)  제거된 가우시안의 원래 xyz
    r_thresh:    float   반경
    Returns: (N,) int64 mask
    """
    if len(removed_xyz) == 0:
        return np.zeros(len(xyz_orig), dtype=np.int64)
    tree = KDTree(removed_xyz)
    dist, _ = tree.query(xyz_orig, k=1)  # 각 남은 점에서 가장 가까운 removed 점까지
    mask = (dist[:, 0] <= r_thresh).astype(np.int64)
    return mask




def smooth_scores(scores, xyz, k_smooth=8):
    """
    각 가우시안 score를 xyz 기준 k-NN 이웃들의 score 평균으로 보정.
    anomaly는 공간적으로 뭉쳐있다는 가정 -> 외톨이 false positive 감소.
    
    scores:   (N,) anomaly score
    xyz:      (N, 3) 정규화 전 원본 xyz
    k_smooth: 이웃 수 (자기 자신 포함)
    Returns:  (N,) smoothed score
    """
    if k_smooth <= 1:
        return scores
    tree = KDTree(xyz)
    # 자기 자신 포함 k개
    _, idx = tree.query(xyz, k=min(k_smooth, len(xyz)))
    return scores[idx].mean(axis=1)



def _load_normal_xyz_raw(data_root, cls):
    """클래스 정상 PLY 1개의 xyz (정규화 전 원본)."""
    import glob as _g
    cands = sorted(_g.glob(os.path.join(data_root, cls, 'normal', 'point_cloud_*.ply')))
    assert len(cands) > 0, f"정상 PLY 없음: {cls}"
    ply = PlyData.read(cands[0])
    v = ply['vertex']
    return np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=1).astype(np.float32)


def compute_density_score(xyz_anom, xyz_normal):
    """
    단방향 density score: 정상 대비 밀도가 낮은 곳만 anomaly로.
    missing 탐지용. burrs(밀도 증가)는 0으로 잘려 중립.
    
    xyz_anom:   (N,3) anomaly PLY xyz (정규화 전 원본)
    xyz_normal: (M,3) 같은 클래스 정상 PLY xyz
    Returns: (N,) score, 높을수록 anomaly
    """
    nt = KDTree(xyz_normal)
    tree = KDTree(xyz_anom)
    knn_d, _ = tree.query(xyz_anom, k=min(13, len(xyz_anom)))
    radius = knn_d[:, 1:].mean() * 1.5
    cnt_a = tree.query_radius(xyz_anom, r=radius, count_only=True).astype(np.float32)
    cnt_n = nt.query_radius(xyz_anom, r=radius, count_only=True).astype(np.float32)
    rel = cnt_a / (cnt_n + 1e-6)
    return np.maximum(0.0, 1.0 - rel)


def minmax_norm(s):
    """per-sample min-max 정규화 -> [0,1]."""
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def fuse_scores(mae_score, density_score, fusion='max', lambda_d=1.0):
    """
    MAE score와 density score를 결합.
    fusion: 'max' | 'sum' | 'mae_only' | 'density_only'
    """
    mae_n = minmax_norm(mae_score)
    den_n = minmax_norm(density_score)
    if fusion == 'mae_only':
        return mae_n
    elif fusion == 'density_only':
        return den_n
    elif fusion == 'sum':
        return mae_n + lambda_d * den_n
    else:  # max
        return np.maximum(mae_n, den_n)



@torch.no_grad()
def compute_anomaly_score(model, params, n_iterations, loss_weights):
    device = params.device
    B, N, D = params.shape
    
    cum_score = torch.zeros(B, N, device=device)
    count = torch.zeros(B, N, device=device)
    
    weight_per_dim = torch.ones(D, device=device)
    weight_per_dim[ATTR_SLICES['xyz']] = loss_weights['xyz']
    weight_per_dim[ATTR_SLICES['f_dc']] = loss_weights['f_dc']
    weight_per_dim[ATTR_SLICES['opacity']] = loss_weights['opacity']
    weight_per_dim[ATTR_SLICES['scale']] = loss_weights['scale']
    weight_per_dim[ATTR_SLICES['rotation']] = loss_weights['rotation']
    
    for it in range(n_iterations):
        out = model(params, return_all=True)
        recon_full = out['recon_full']
        target_full = out['target_full']
        mask = out['mask']
        patch_idx = out['patch_idx']
        
        diff_sq = (recon_full - target_full) ** 2
        diff_weighted = (diff_sq * weight_per_dim).sum(dim=-1)
        
        for b in range(B):
            mask_b = mask[b]
            idx_masked = patch_idx[b][mask_b]
            score_masked = diff_weighted[b][mask_b]
            
            idx_flat = idx_masked.reshape(-1)
            score_flat = score_masked.reshape(-1)
            
            cum_score[b].scatter_add_(0, idx_flat, score_flat)
            count[b].scatter_add_(0, idx_flat, torch.ones_like(score_flat))
    
    avg_score = cum_score / count.clamp(min=1)
    return avg_score


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def evaluate(cfg, ckpt_path, normalizer_path, do_subsample=False, save_results=True, k_radius=2.5, k_smooth=8, fusion="max", lambda_d=1.0):
    device = cfg.train.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Subsample: {'O' if do_subsample else 'X (원본 크기 그대로 사용)'}")
    
    normalizer = GaussianNormalizer.load(normalizer_path)
    
    print(f"\n체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = GaussianMAE(
        num_groups=cfg.model.num_groups,
        group_size=cfg.model.group_size,
        input_dim=cfg.data.input_dim,
        embed_dim=cfg.model.embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        decoder_depth=cfg.model.decoder_depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        mask_ratio=cfg.model.mask_ratio,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    loss_weights = {
        'xyz':      cfg.model.loss_weight_xyz,
        'f_dc':     cfg.model.loss_weight_f_dc,
        'opacity':  cfg.model.loss_weight_opacity,
        'scale':    cfg.model.loss_weight_scale,
        'rotation': cfg.model.loss_weight_rotation,
    }
    
    anom_ds = AnomalyGaussianDataset(
        data_root=cfg.data.data_root,
        classes=cfg.data.classes,
        anomaly_types=cfg.data.anomaly_types,
        n_gaussians=cfg.data.n_gaussians,
        normalizer=normalizer,
        do_subsample=do_subsample,
    )
    
    # 클래스별 missing GT 반경 r_thresh 사전 계산
    class_radii = precompute_class_radii(cfg.data.classes, cfg.data.data_root, k_radius=k_radius)
    
    # hybrid용: 클래스별 정상 xyz 캐싱
    print(f'[Hybrid] fusion={fusion}, lambda_d={lambda_d} / 정상 xyz 캐싱 중...')
    class_normal_xyz = {c: _load_normal_xyz_raw(cfg.data.data_root, c) for c in cfg.data.classes}
    anom_loader = DataLoader(anom_ds, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    results = {}
    
    print(f"\n[Anomaly 평가] {len(anom_ds)} 샘플, iter={cfg.eval.n_iterations}회")
    
    for i, batch in enumerate(anom_loader):
        params = batch['params'].to(device, non_blocking=True)
        gt = batch['gt_mask'].squeeze(0).cpu().numpy()
        cls = batch['class_name'][0]
        atype = batch['anomaly_type'][0]
        
        # missing은 GT 재계산 (placeholder가 아닌 NN 기반)
        if atype == 'missing_recon':
            xyz_orig = batch['xyz_orig'].squeeze(0).cpu().numpy()
            removed_xyz = batch['removed_xyz'].squeeze(0).cpu().numpy()
            gt = compute_missing_gt(xyz_orig, removed_xyz, class_radii[cls])
        
        mae_scores = compute_anomaly_score(model, params, cfg.eval.n_iterations, loss_weights)
        mae_scores = mae_scores.squeeze(0).cpu().numpy()
        
        # score smoothing (이웃 평균) — 후처리
        xyz_orig_np = batch['xyz_orig'].squeeze(0).cpu().numpy()
        if k_smooth > 1:
            mae_scores = smooth_scores(mae_scores, xyz_orig_np, k_smooth=k_smooth)
        
        # density score (정상 대비 밀도)
        density_sc = compute_density_score(xyz_orig_np, class_normal_xyz[cls])
        # fusion된 최종 score (요청된 fusion 방식)
        scores = fuse_scores(mae_scores, density_sc, fusion=fusion, lambda_d=lambda_d)
        
        if cls not in results:
            results[cls] = {}
        if atype not in results[cls]:
            results[cls][atype] = {'gt': [], 'score': [], 'mae': [], 'density': []}
        
        results[cls][atype]['gt'].append(gt)
        results[cls][atype]['score'].append(scores)
        results[cls][atype]['mae'].append(mae_scores)
        results[cls][atype]['density'].append(density_sc)
        
        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{len(anom_ds)}")
    

    print(f"\n{'=' * 70}")
    print(f"Gaussian-level AUROC")
    print(f"{'=' * 70}")
    
    # Overall AUROC
    all_gt, all_score = [], []
    for cls in results:
        for atype in results[cls]:
            for gt, sc in zip(results[cls][atype]['gt'], results[cls][atype]['score']):
                all_gt.append(gt)
                all_score.append(sc)
    
    all_gt = np.concatenate(all_gt)
    all_score = np.concatenate(all_score)
    overall_auc = safe_auroc(all_gt, all_score)
    print(f"\nOverall AUROC: {overall_auc:.4f}")
    
    # 타입별 AUROC
    print(f"\n[타입별 AUROC (모든 클래스 통합)]")
    type_aucs = {}
    for atype in cfg.data.anomaly_types:
        gts, scs = [], []
        for cls in results:
            if atype in results[cls]:
                for gt, sc in zip(results[cls][atype]['gt'], results[cls][atype]['score']):
                    gts.append(gt)
                    scs.append(sc)
        if gts:
            gts = np.concatenate(gts)
            scs = np.concatenate(scs)
            auc = safe_auroc(gts, scs)
            type_aucs[atype] = auc
            print(f"  {atype:20s}: {auc:.4f}" if auc else f"  {atype:20s}: N/A")
    
    # 클래스별 AUROC
    print(f"\n[클래스별 AUROC (모든 타입 통합)]")
    cls_aucs = {}
    for cls in sorted(results.keys()):
        gts, scs = [], []
        for atype in results[cls]:
            for gt, sc in zip(results[cls][atype]['gt'], results[cls][atype]['score']):
                gts.append(gt)
                scs.append(sc)
        gts = np.concatenate(gts)
        scs = np.concatenate(scs)
        auc = safe_auroc(gts, scs)
        cls_aucs[cls] = auc
        print(f"  {cls:15s}: {auc:.4f}" if auc else f"  {cls:15s}: N/A")
    
    # 클래스별 × 타입별 AUROC (가장 자세한 분석)
    print(f"\n[클래스별 + 타입별 AUROC]")
    
    type_short = {
        'burrs_recon': 'burrs',
        'stains_recon': 'stains',
        'missing_recon': 'missing',
    }
    
    # 헤더
    header = f"  {'Class':<15s}"
    for atype in cfg.data.anomaly_types:
        short = type_short.get(atype, atype)
        header += f"{short:>10s}"
    header += f"{'avg':>10s}"
    print(header)
    print(f"  {'-' * 15}{'-' * 40}")
    
    # 본체
    cls_type_aucs = {}  # cls_type_aucs[cls][atype] = auc
    for cls in sorted(results.keys()):
        cls_type_aucs[cls] = {}
        line = f"  {cls:<15s}"
        cls_aucs_list = []
        for atype in cfg.data.anomaly_types:
            if atype in results[cls]:
                gts, scs = [], []
                for gt, sc in zip(results[cls][atype]['gt'], results[cls][atype]['score']):
                    gts.append(gt)
                    scs.append(sc)
                gts = np.concatenate(gts)
                scs = np.concatenate(scs)
                auc = safe_auroc(gts, scs)
                cls_type_aucs[cls][atype] = auc
                if auc is not None:
                    line += f"{auc:>10.4f}"
                    cls_aucs_list.append(auc)
                else:
                    line += f"{'N/A':>10s}"
            else:
                line += f"{'-':>10s}"
        # 클래스 평균
        if cls_aucs_list:
            avg = np.mean(cls_aucs_list)
            line += f"{avg:>10.4f}"
        print(line)
    
    # 타입별 통계 (mean/std/min/max)
    print(f"\n[타입별 통계 (across classes)]")
    print(f"  {'Type':<20s}{'mean':>10s}{'std':>10s}{'min':>10s}{'max':>10s}")
    print(f"  {'-' * 60}")
    for atype in cfg.data.anomaly_types:
        type_class_aucs = []
        for cls in cls_type_aucs:
            if atype in cls_type_aucs[cls] and cls_type_aucs[cls][atype] is not None:
                type_class_aucs.append(cls_type_aucs[cls][atype])
        if type_class_aucs:
            type_class_aucs = np.array(type_class_aucs)
            print(f"  {atype:<20s}{type_class_aucs.mean():>10.4f}"
                  f"{type_class_aucs.std():>10.4f}"
                  f"{type_class_aucs.min():>10.4f}"
                  f"{type_class_aucs.max():>10.4f}")

    if save_results:
        # 결과 저장 디렉토리
        output_dir = os.path.dirname(ckpt_path)
        result_dir = os.path.join(output_dir, 'eval_results')
        os.makedirs(result_dir, exist_ok=True)
        
        # JSON으로 AUROC 결과 저장
        result_summary = {
            'overall_auroc': float(overall_auc),
            'type_aucs': {k: float(v) if v is not None else None for k, v in type_aucs.items()},
            'cls_aucs': {k: float(v) if v is not None else None for k, v in cls_aucs.items()},
            'cls_type_aucs': {
                cls: {atype: float(auc) if auc is not None else None 
                      for atype, auc in atypes.items()}
                for cls, atypes in cls_type_aucs.items()
            },
            'config': {
                'do_subsample': do_subsample,
                'n_gaussians': cfg.data.n_gaussians,
                'n_iterations': cfg.eval.n_iterations,
                'classes': cfg.data.classes,
                'anomaly_types': cfg.data.anomaly_types,
                'k_radius': k_radius,
                'k_smooth': k_smooth,
                'fusion': fusion,
                'lambda_d': lambda_d,
            }
        }
        
        suffix = 'subsample' if do_subsample else 'orig'
        json_path = os.path.join(result_dir, f'eval_summary_{suffix}.json')
        with open(json_path, 'w') as f:
            json.dump(result_summary, f, indent=2)
        print(f"\n 결과 저장: {json_path}")
        
        # raw 데이터 (gt + score) 저장 (재분석 가능)
        raw_data = {
            'classes': [],
            'anomaly_types': [],
            'gts': [],
            'scores': [],
            'mae': [],
            'density': [],
        }
        for cls in sorted(results.keys()):
            for atype in cfg.data.anomaly_types:
                if atype in results[cls]:
                    for gt, sc, mae_s, den_s in zip(
                            results[cls][atype]['gt'], results[cls][atype]['score'],
                            results[cls][atype]['mae'], results[cls][atype]['density']):
                        raw_data['classes'].append(cls)
                        raw_data['anomaly_types'].append(atype)
                        raw_data['gts'].append(gt)
                        raw_data['scores'].append(sc)
                        raw_data['mae'].append(mae_s)
                        raw_data['density'].append(den_s)
        
        npz_path = os.path.join(result_dir, f'eval_raw_{suffix}.npz')
        np.savez(npz_path,
                 classes=np.array(raw_data['classes']),
                 anomaly_types=np.array(raw_data['anomaly_types']),
                 gts=np.array(raw_data['gts'], dtype=object),
                 scores=np.array(raw_data['scores'], dtype=object),
                 mae=np.array(raw_data['mae'], dtype=object),
                 density=np.array(raw_data['density'], dtype=object))
        print(f" Raw 결과 저장: {npz_path}")
    
    return overall_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--normalizer', type=str, required=True)
    parser.add_argument('--n_gaussians', type=int, default=None)
    parser.add_argument('--n_iter', type=int, default=None)
    parser.add_argument('--classes', nargs='+', default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--anomaly_types', nargs='+', default=None)
    parser.add_argument('--subsample', action='store_true',
                        help='subsample 사용 (default: 원본 크기)')
    parser.add_argument('--no_save', action='store_true',
                        help='결과 저장 안 함')
    parser.add_argument('--k_radius', type=float, default=2.5,
                        help='Missing GT 반경 = 정상 NN 평균 × k_radius (기본 2.5)')
    parser.add_argument('--k_smooth', type=int, default=8,
                        help='Score smoothing 이웃 수 (1이면 smoothing 없음, 기본 8)')
    parser.add_argument('--fusion', type=str, default='max',
                        choices=['max', 'sum', 'mae_only', 'density_only'],
                        help='MAE score와 density score 결합 방식 (기본 max)')
    parser.add_argument('--lambda_d', type=float, default=1.0,
                        help='sum fusion 시 density 가중치 (기본 1.0)')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Transformer embedding dim (학습과 동일하게)')
    parser.add_argument('--encoder_depth', type=int, default=None)
    parser.add_argument('--decoder_depth', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    args = parser.parse_args()
    
    cfg = Config()
    if args.n_gaussians: cfg.data.n_gaussians = args.n_gaussians
    if args.n_iter: cfg.eval.n_iterations = args.n_iter
    if args.classes: cfg.data.classes = args.classes
    if args.data_root: cfg.data.data_root = args.data_root
    if args.anomaly_types: cfg.data.anomaly_types = args.anomaly_types
    
    # 모델 크기 (학습 시와 동일해야 함)
    if args.embed_dim: cfg.model.embed_dim = args.embed_dim
    if args.encoder_depth: cfg.model.encoder_depth = args.encoder_depth
    if args.decoder_depth: cfg.model.decoder_depth = args.decoder_depth
    if args.num_heads: cfg.model.num_heads = args.num_heads
    
    print(f"Model: embed={cfg.model.embed_dim}, "
          f"enc={cfg.model.encoder_depth}, dec={cfg.model.decoder_depth}, "
          f"heads={cfg.model.num_heads}")
    
    print(cfg)
    evaluate(cfg, args.ckpt, args.normalizer, 
             do_subsample=args.subsample,
             save_results=not args.no_save,
             k_radius=args.k_radius,
             k_smooth=args.k_smooth,
             fusion=args.fusion,
             lambda_d=args.lambda_d)