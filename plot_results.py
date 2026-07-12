"""
plot_results.py
===============
Reads the metrics.csv files written by ppo_continuous.py and produces:
  1. Per-environment learning curves (episodic return vs timestep),
     averaged over seeds, with mean +/- std shading.
  2. An ablation figure overlaying clip / noclip / kl on one environment.

Directory layout expected (default from ppo_continuous.py):
    runs/<env>__<mode>__seed<seed>__<timestamp>/metrics.csv

Usage:
    python plot_results.py --log-dir runs --out figures
    python plot_results.py --log-dir runs --out figures --env HalfCheetah-v5
"""

import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_run_name(path):
    """runs/<env>__<mode>__seed<seed>__<ts>/metrics.csv -> (env, mode, seed)."""
    name = os.path.basename(os.path.dirname(path))
    parts = name.split("__")
    env = parts[0]
    mode = parts[1]
    seed = int(parts[2].replace("seed", ""))
    return env, mode, seed


def load_returns(path):
    """Return (steps, returns) arrays for episodic_return rows in one CSV."""
    df = pd.read_csv(path)
    df = df[df["metric"] == "episodic_return"]
    return df["global_step"].values, df["value"].values


def bin_and_average(runs, num_bins=100, max_step=None):
    """
    runs: list of (steps, values) tuples (one per seed).
    Interpolate each run onto a common grid, then return grid, mean, std.
    """
    if max_step is None:
        max_step = min(r[0].max() for r in runs if len(r[0]) > 0)
    grid = np.linspace(0, max_step, num_bins)
    interp = []
    for steps, vals in runs:
        if len(steps) < 2:
            continue
        order = np.argsort(steps)
        steps, vals = steps[order], vals[order]
        interp.append(np.interp(grid, steps, vals))
    if not interp:
        return grid, np.zeros_like(grid), np.zeros_like(grid)
    stack = np.vstack(interp)
    return grid, stack.mean(axis=0), stack.std(axis=0)


def smooth(y, window=5):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def collect(log_dir):
    """Return nested dict: data[env][mode] = list of (steps, returns) per seed."""
    data = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(log_dir, "*", "metrics.csv")):
        try:
            env, mode, seed = parse_run_name(path)
            steps, rets = load_returns(path)
            if len(steps) > 0:
                data[env][mode].append((steps, rets))
        except Exception as e:
            print(f"[skip] {path}: {e}")
    return data


def plot_learning_curves(data, out_dir):
    """One figure per environment, one line per mode, seed-averaged with shading."""
    os.makedirs(out_dir, exist_ok=True)
    colors = {"clip": "C0", "noclip": "C3", "kl": "C2"}
    labels = {"clip": "PPO (clip, e=0.2)", "noclip": "No clipping", "kl": "Adaptive KL"}

    for env, modes in data.items():
        plt.figure(figsize=(7, 4.5))
        for mode, runs in sorted(modes.items()):
            grid, mean, std = bin_and_average(runs)
            mean_s = smooth(mean); std_s = smooth(std)
            plt.plot(grid, mean_s, label=f"{labels.get(mode, mode)} (n={len(runs)})",
                     color=colors.get(mode))
            plt.fill_between(grid, mean_s - std_s, mean_s + std_s,
                             alpha=0.2, color=colors.get(mode))
        plt.xlabel("Timestep")
        plt.ylabel("Episodic return")
        plt.title(f"PPO on {env}")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fp = os.path.join(out_dir, f"curve_{env}.png")
        plt.savefig(fp, dpi=150)
        plt.close()
        print(f"wrote {fp}")


def plot_ablation_bars(data, out_dir):
    """
    Bar chart of final performance (mean over last 10% of training, then over
    seeds) per mode, per environment -- analogous to the paper's Table 1.
    """
    os.makedirs(out_dir, exist_ok=True)
    envs = sorted(data.keys())
    modes = ["clip", "noclip", "kl"]
    means = defaultdict(list)
    errs = defaultdict(list)

    for env in envs:
        for mode in modes:
            runs = data[env].get(mode, [])
            finals = []
            for steps, rets in runs:
                if len(rets) == 0:
                    continue
                k = max(1, len(rets) // 10)
                finals.append(np.mean(rets[-k:]))
            if finals:
                means[mode].append(np.mean(finals))
                errs[mode].append(np.std(finals))
            else:
                means[mode].append(0.0)
                errs[mode].append(0.0)

    x = np.arange(len(envs))
    width = 0.25
    plt.figure(figsize=(8, 4.5))
    for i, mode in enumerate(modes):
        plt.bar(x + (i - 1) * width, means[mode], width,
                yerr=errs[mode], capsize=3, label=mode)
    plt.xticks(x, envs, rotation=20, ha="right")
    plt.ylabel("Final episodic return (last 10%)")
    plt.title("Ablation: clipped vs no-clip vs KL penalty")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fp = os.path.join(out_dir, "ablation_bars.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"wrote {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--env", default=None, help="restrict to one env id")
    args = ap.parse_args()

    data = collect(args.log_dir)
    if args.env:
        data = {args.env: data[args.env]} if args.env in data else {}
    if not data:
        print("No data found. Check --log-dir.")
        return

    plot_learning_curves(data, args.out)
    plot_ablation_bars(data, args.out)


if __name__ == "__main__":
    main()
