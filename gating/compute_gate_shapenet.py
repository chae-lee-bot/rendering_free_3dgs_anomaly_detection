"""ShapeNet Hybrid gating + per-gaussian AUROC."""
import os
import numpy as np
from sklearn.metrics import roc_auc_score

OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT', './output')
NPZ = f"{OUTPUT_ROOT}/shapenet/eval_results/eval_raw.npz"
DEN = f"{OUTPUT_ROOT}/shapenet/eval_results/density_bidir.npz"

d = np.load(NPZ, allow_pickle=True)
atype = [str(x) for x in d['anomaly_types']]
gts   = [np.asarray(x) for x in d['gts']]
maes  = [np.asarray(x, dtype=np.float64) for x in d['mae']]
rels  = [np.asarray(x, dtype=np.float64) for x in np.load(DEN, allow_pickle=True)['rel']]
N = len(gts)
print(f"[ShapeNet in-domain] 샘플 {N}개\n")

def minmax(s):
    lo,hi = float(s.min()), float(s.max())
    return np.zeros_like(s) if hi-lo<1e-12 else (s-lo)/(hi-lo)
def safe_auc(g,s):
    return roc_auc_score(g,s) if len(np.unique(g))>1 else None

ATYPES = ['burrs_recon','stains_recon','missing_recon']
mae_s = [minmax(m) for m in maes]
den_raw = [np.abs(1.0-r) for r in rels]
den_s = [minmax(x) for x in den_raw]

struct = np.array([np.quantile(den_raw[i],0.95) for i in range(N)])
recon  = np.array([np.quantile(maes[i],0.95) for i in range(N)])
ratio  = struct/(recon+1e-9)

concentration = np.array([
    np.sort(den_raw[i])[::-1][:max(1,len(den_raw[i])//20)].sum() / (den_raw[i].sum()+1e-9)
    for i in range(N)
])
sharpness = np.array([
    np.quantile(den_raw[i],0.90) / (np.quantile(den_raw[i],0.50)+1e-9)
    for i in range(N)
])

is_struct = np.array([atype[i]!='stains_recon' for i in range(N)]).astype(int)
print("[게이팅 지표 AUROC (구조이상 vs stains)]")
print(f"  ratio              : {roc_auc_score(is_struct,ratio):.4f}")
print(f"  concentration      : {roc_auc_score(is_struct,concentration):.4f}")
print(f"  sharpness          : {roc_auc_score(is_struct,sharpness):.4f}")
def zn(x): return (x-x.mean())/(x.std()+1e-9)
combo = zn(ratio) + zn(concentration)
combo2 = zn(ratio) + zn(concentration) + zn(sharpness)
print(f"  ratio+concentration: {roc_auc_score(is_struct,combo):.4f}")
print(f"  ratio+conc+sharp   : {roc_auc_score(is_struct,combo2):.4f}\n")

def report(label, score_list):
    pt = {a:[] for a in ATYPES}
    for i in range(N):
        a = safe_auc(gts[i], score_list[i])
        if a is not None: pt[atype[i]].append(a)
    tm = [np.mean(pt[a]) if pt[a] else float('nan') for a in ATYPES]
    allv = sum(pt.values(),[])
    print(f"{label:30s} {np.mean(allv):>8.4f}" + "".join(f" {m:>9.4f}" for m in tm), flush=True)

print(f"{'strategy':30s} {'Overall':>8s} {'burrs':>9s} {'stains':>9s} {'missing':>9s}")
print("-"*70)
report("mae_only", mae_s)
report("density_only", den_s)

for name, metric in [('ratio',ratio),('combo(r+c)',combo),('combo(r+c+s)',combo2)]:
    for q in [0.35,0.4,0.45,0.5]:
        thr = np.quantile(metric,q)
        report(f"{name} gate q={q}",
               [den_s[i] if metric[i]>thr else mae_s[i] for i in range(N)])

report("ORACLE (참고)",
       [den_s[i] if atype[i]=='missing_recon' else mae_s[i] for i in range(N)])
