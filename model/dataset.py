"""
전체는 14차원 = xyz(3) + f_dc(3) + opacity(1) + scale(3) + rotation(4)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData


ATTR_NAMES = [
    'x', 'y', 'z',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
]

ATTR_SLICES = {
    'xyz':      slice(0, 3),
    'f_dc':     slice(3, 6),
    'opacity':  slice(6, 7),
    'scale':    slice(7, 10),
    'rotation': slice(10, 14),
}


def load_ply_14dim(ply_path):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    arrays = []
    for name in ATTR_NAMES:
        arrays.append(np.asarray(v[name], dtype=np.float32))
    params = np.stack(arrays, axis=1)
    return params


def subsample_or_pad(params, target_n, seed=None):
    n = len(params)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    if n == target_n:
        return params, np.arange(n)
    elif n > target_n:
        idx = rng.choice(n, target_n, replace=False)
    else:
        idx = rng.choice(n, target_n, replace=True)
    
    return params[idx], idx



class GaussianNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def fit(self, all_params):
        self.mean = all_params.mean(axis=0).astype(np.float32)
        self.std = all_params.std(axis=0).astype(np.float32)
        self.std = np.maximum(self.std, 1e-6)
    
    def normalize(self, params):
        if isinstance(params, np.ndarray):
            return (params - self.mean) / self.std
        elif isinstance(params, torch.Tensor):
            mean_t = torch.from_numpy(self.mean).to(params.device)
            std_t = torch.from_numpy(self.std).to(params.device)
            return (params - mean_t) / std_t
    
    def denormalize(self, params_norm):
        if isinstance(params_norm, np.ndarray):
            return params_norm * self.std + self.mean
        elif isinstance(params_norm, torch.Tensor):
            mean_t = torch.from_numpy(self.mean).to(params_norm.device)
            std_t = torch.from_numpy(self.std).to(params_norm.device)
            return params_norm * std_t + mean_t
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
        print(f"Normalizer 저장: {path}")
    
    @classmethod
    def load(cls, path):
        data = np.load(path)
        norm = cls(mean=data['mean'], std=data['std'])
        print(f"Normalizer 로드: {path}")
        return norm
    
    def __repr__(self):
        if self.mean is None:
            return "GaussianNormalizer (not fitted)"
        s = "GaussianNormalizer:\n"
        for name, sl in ATTR_SLICES.items():
            m = self.mean[sl]
            st = self.std[sl]
            s += f"  {name}: mean={[f'{x:.3f}' for x in m]}, std={[f'{x:.3f}' for x in st]}\n"
        return s


def compute_normalizer(data_root, classes, n_samples_per_ply=5000, seed=42):
    print(f"[Normalizer] 통계 계산 중...")
    rng = np.random.RandomState(seed)
    
    all_samples = []
    n_total_plys = 0
    for cls in classes:
        normal_dir = os.path.join(data_root, cls, 'normal')
        ply_files = sorted(glob.glob(os.path.join(normal_dir, 'point_cloud_*.ply')))
        for p in ply_files:
            params = load_ply_14dim(p)
            n = len(params)
            sample_n = min(n_samples_per_ply, n)
            idx = rng.choice(n, sample_n, replace=False)
            all_samples.append(params[idx])
            n_total_plys += 1
    
    all_samples = np.concatenate(all_samples, axis=0)
    print(f"  총 {n_total_plys}개 PLY, {len(all_samples):,}개 샘플로 통계 계산")
    
    normalizer = GaussianNormalizer()
    normalizer.fit(all_samples)
    print(normalizer)
    return normalizer



class NormalGaussianDataset(Dataset):
    def __init__(self, data_root, classes, n_gaussians=20000, normalizer=None, 
                 do_subsample=True, seed=42):
        self.data_root = data_root
        self.n_gaussians = n_gaussians
        self.normalizer = normalizer
        self.do_subsample = do_subsample
        self.seed = seed
        
        self.samples = []
        for cls in classes:
            normal_dir = os.path.join(data_root, cls, 'normal')
            ply_files = sorted(glob.glob(os.path.join(normal_dir, 'point_cloud_*.ply')))
            for p in ply_files:
                self.samples.append((p, cls))
        
        print(f"[NormalGaussianDataset] 총 {len(self.samples)}개 샘플 ({len(classes)}개 클래스)"
              f" [정규화: {'O' if normalizer is not None else 'X'},"
              f" subsample: {'O' if do_subsample else 'X (원본 크기)'}]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ply_path, cls = self.samples[idx]
        params = load_ply_14dim(ply_path)
        
        if self.do_subsample and len(params) != self.n_gaussians:
            params, _ = subsample_or_pad(params, self.n_gaussians)
        
        if self.normalizer is not None:
            params = self.normalizer.normalize(params)
        
        return {
            'params': torch.from_numpy(params).float(),
            'class_name': cls,
            'ply_path': ply_path,
        }


class AnomalyGaussianDataset(Dataset):
    """
    평가용 dataset (Anomaly PLY + GT mask).
    do_subsample=True (기본): n_gaussians로 통일
    do_subsample=False: 원본 크기 그대로 (정확한 평가용)
    """
    def __init__(self, data_root, classes, anomaly_types, n_gaussians=20000, 
                 normalizer=None, do_subsample=True, seed=42):
        self.data_root = data_root
        self.n_gaussians = n_gaussians
        self.normalizer = normalizer
        self.do_subsample = do_subsample
        self.seed = seed
        
        self.samples = []
        for cls in classes:
            for atype in anomaly_types:
                anom_dir = os.path.join(data_root, cls, 'anomaly', atype)
                if not os.path.exists(anom_dir):
                    continue
                random_dirs = sorted(glob.glob(os.path.join(anom_dir, 'random_*')))
                for rdir in random_dirs:
                    ply = os.path.join(rdir, 'point_cloud.ply')
                    gt = os.path.join(rdir, 'gt_mask.npy')
                    if os.path.exists(ply) and os.path.exists(gt):
                        random_idx = int(os.path.basename(rdir).replace('random_', ''))
                        removed_xyz_path = os.path.join(rdir, 'removed_xyz.npy')
                        has_removed = os.path.exists(removed_xyz_path)
                        self.samples.append({
                            'ply_path': ply,
                            'gt_path': gt,
                            'class_name': cls,
                            'anomaly_type': atype,
                            'random_idx': random_idx,
                            'removed_xyz_path': removed_xyz_path if has_removed else '',
                        })
        
        print(f"[AnomalyGaussianDataset] 총 {len(self.samples)}개 샘플 "
              f"({len(classes)} 클래스 × {len(anomaly_types)} 타입)"
              f" [정규화: {'O' if normalizer is not None else 'X'},"
              f" subsample: {'O' if do_subsample else 'X (원본 크기)'}]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        params = load_ply_14dim(s['ply_path'])
        gt_mask = np.load(s['gt_path']).astype(np.int64)
        
        assert len(params) == len(gt_mask)
        
        if self.do_subsample and len(params) != self.n_gaussians:
            params, idx_kept = subsample_or_pad(params, self.n_gaussians)
            gt_mask = gt_mask[idx_kept]
        
        xyz_orig = params[:, 0:3].copy()
        
        if self.normalizer is not None:
            params = self.normalizer.normalize(params)
        
        if s['removed_xyz_path']:
            removed_xyz = np.load(s['removed_xyz_path']).astype(np.float32)
        else:
            removed_xyz = np.zeros((0, 3), dtype=np.float32)
        
        return {
            'params': torch.from_numpy(params).float(),
            'gt_mask': torch.from_numpy(gt_mask).long(),
            'xyz_orig': torch.from_numpy(xyz_orig).float(),
            'removed_xyz': torch.from_numpy(removed_xyz).float(),
            'class_name': s['class_name'],
            'anomaly_type': s['anomaly_type'],
            'random_idx': s['random_idx'],
            'ply_path': s['ply_path'],
        }



if __name__ == "__main__":
    from configs import Config
    cfg = Config()
    
    normalizer = compute_normalizer(
        data_root=cfg.data.data_root,
        classes=['01Gorilla'],
    )
    
    print("\n" + "=" * 60)
    print("Subsample O (학습용)")
    print("=" * 60)
    
    ds_sub = NormalGaussianDataset(
        data_root=cfg.data.data_root,
        classes=['01Gorilla'],
        n_gaussians=cfg.data.n_gaussians,
        normalizer=normalizer,
        do_subsample=True,
    )
    s = ds_sub[0]
    print(f"샘플 0 shape: {s['params'].shape}  (subsample 적용)")
    
    print("\n" + "=" * 60)
    print("Subsample X (평가용)")
    print("=" * 60)
    
    ds_full = NormalGaussianDataset(
        data_root=cfg.data.data_root,
        classes=['01Gorilla'],
        n_gaussians=cfg.data.n_gaussians,
        normalizer=normalizer,
        do_subsample=False,
    )
    s = ds_full[0]
    print(f"샘플 0 shape: {s['params'].shape}  (원본 크기)")
    