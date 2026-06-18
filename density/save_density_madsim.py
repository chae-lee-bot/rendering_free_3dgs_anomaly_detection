"""MAD-Sim KDTree density (bidirectional)."""
import os, numpy as np, glob, time
from plyfile import PlyData
from sklearn.neighbors import KDTree

DATA_ROOT   = os.environ.get('DATA_ROOT', '.')
OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT', './output')

NORMAL = f"{DATA_ROOT}/MAD_SIM_FINAL"
ANOM   = f"{DATA_ROOT}/MAD_SIM_FINAL"
NPZ    = f"{OUTPUT_ROOT}/madsim/eval_results/eval_raw.npz"
OUT    = f"{OUTPUT_ROOT}/madsim/eval_results/density_bidir.npz"

def load_xyz(path):
    v = PlyData.read(path)['vertex']
    return np.stack([v['x'],v['y'],v['z']],axis=1).astype(np.float32)

print("npz 로딩...", flush=True)
d = np.load(NPZ, allow_pickle=True)
npz_cls   = [str(x) for x in d['classes']]
npz_atype = [str(x) for x in d['anomaly_types']]
npz_gt    = [np.asarray(x) for x in d['gts']]
print(f"  샘플 {len(npz_cls)}개\n", flush=True)

norm_cache = {}
def get_nt(cls):
    if cls not in norm_cache:
        nx = load_xyz(os.path.join(NORMAL,cls,'normal','point_cloud_0.ply'))
        norm_cache[cls] = KDTree(nx)
    return norm_cache[cls]

def density_bidir(xyz, nt):
    tree = KDTree(xyz)
    knn_d,_ = tree.query(xyz, k=13)
    radius = knn_d[:,1:].mean()*1.5
    cnt_a = tree.query_radius(xyz, r=radius, count_only=True).astype(np.float64)
    cnt_n = nt.query_radius(xyz, r=radius, count_only=True).astype(np.float64)
    rel = cnt_a/(cnt_n+1e-6)
    return rel   # rel 자체 저장 (나중에 |1-rel| 등 자유롭게)

t0 = time.time()
rdir_cache = {}; seen = {}
rels = []
for i in range(len(npz_cls)):
    cls, atype = npz_cls[i], npz_atype[i]
    key = (cls, atype)
    if key not in rdir_cache:
        rdir_cache[key] = sorted(glob.glob(os.path.join(ANOM, cls, atype, 'random_*')))
        seen[key] = 0
    nt = get_nt(cls)
    matched = None
    while seen[key] < len(rdir_cache[key]):
        rdir = rdir_cache[key][seen[key]]; seen[key] += 1
        xyz = load_xyz(os.path.join(rdir,'point_cloud.ply'))
        if len(xyz) == len(npz_gt[i]):
            matched = (xyz, nt); break
    if matched is None:
        raise RuntimeError(f"매칭 실패 idx {i} {key}")
    xyz, nt = matched
    rels.append(density_bidir(xyz, nt))
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(npz_cls)}  ({time.time()-t0:.0f}s)", flush=True)

np.savez(OUT, rel=np.array(rels, dtype=object))
print(f"\n저장 완료: {OUT}  ({time.time()-t0:.0f}s)", flush=True)
