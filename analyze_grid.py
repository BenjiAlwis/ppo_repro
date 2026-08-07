import os, glob, argparse
from collections import defaultdict
import numpy as np, pandas as pd

def parse_run_name(path):
    name = os.path.basename(os.path.dirname(path))
    env, mode, seed = name.split("__")[:3]
    return env, mode, int(seed.replace("seed", ""))

def final_return(path, last_n):
    df = pd.read_csv(path)
    r = df[df["metric"] == "episodic_return"]["value"].values
    if len(r) == 0: return np.nan
    return float(np.mean(r[-min(last_n, len(r)):]))

def mean_kl(path):
    df = pd.read_csv(path)
    k = df[df["metric"] == "approx_kl"]["value"].values
    return float(np.mean(k)) if len(k) else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="/workspace/runs_1M")
    ap.add_argument("--last-n", type=int, default=20)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    data = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(a.log_dir, "*", "metrics.csv")):
        env, mode, seed = parse_run_name(p)
        data[env][mode].append((seed, final_return(p, a.last_n), mean_kl(p)))
    modes = ["clip", "noclip", "kl"]; rows = []
    print("\n" + "="*78)
    print(f"PPO ABLATION SUMMARY (final return = mean of last {a.last_n} episodes)")
    print("="*78)
    for env in sorted(data):
        print(f"\n### {env}")
        print(f"{'mode':<8}{'seeds':>6}{'mean':>10}{'std':>9}{'min':>10}{'max':>10}{'mean_KL':>11}")
        print("-"*64)
        for m in modes:
            runs = data[env].get(m, [])
            if not runs: continue
            rets = np.array([r for _,r,_ in runs]); kls = np.array([k for _,_,k in runs])
            rets = rets[~np.isnan(rets)]; kls = kls[~np.isnan(kls)]
            if len(rets)==0: continue
            print(f"{m:<8}{len(rets):>6}{rets.mean():>10.1f}{rets.std():>9.1f}{rets.min():>10.1f}{rets.max():>10.1f}{np.nanmean(kls):>11.4f}")
            rows.append({"env":env,"mode":m,"n_seeds":len(rets),"return_mean":round(rets.mean(),1),
                         "return_std":round(rets.std(),1),"return_min":round(rets.min(),1),
                         "return_max":round(rets.max(),1),"mean_kl":round(float(np.nanmean(kls)),4)})
    print("\n" + "="*78)
    print("KL DIVERGENCE PER UPDATE (clip vs noclip vs kl)")
    print("="*78)
    print(f"{'env':<26}{'clip':>10}{'noclip':>12}{'kl':>10}{'noclip/clip':>14}")
    print("-"*72)
    for env in sorted(data):
        kv = {}
        for m in modes:
            kls = np.array([k for _,_,k in data[env].get(m,[])])
            kls = kls[~np.isnan(kls)]
            kv[m] = float(np.mean(kls)) if len(kls) else np.nan
        c,nc,k = kv.get("clip",np.nan),kv.get("noclip",np.nan),kv.get("kl",np.nan)
        ratio = nc/c if (c and not np.isnan(c) and not np.isnan(nc)) else np.nan
        print(f"{env:<26}{c:>10.4f}{nc:>12.4f}{k:>10.4f}{ratio:>13.1f}x")
    if a.csv:
        pd.DataFrame(rows).to_csv(a.csv, index=False)
        print(f"\nwrote summary -> {a.csv}")

if __name__ == "__main__":
    main()
