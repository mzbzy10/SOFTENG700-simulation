"""Generate poster figures (SVG) from the saved experiment JSONs.

Run from the repo root:  python poster/make_figs.py
Outputs to poster/figs/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "poster", "figs")
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "v1": "results/v1_dqn_minrate/transfer_20260904_002751.json",
    "v2": "results/v2_ddqn_servedratio/transfer_20260909_195337.json",
    "v3": "results/v3_ddqn_long/transfer_20260910_020846.json",
}
DATA = {k: json.load(open(os.path.join(ROOT, p))) for k, p in RUNS.items()}
BASE = json.load(open(os.path.join(ROOT, "results/baselines.json")))

ENVS = ["balanced", "embb_intensive", "urllc_intensive", "mmtc_intensive"]
LAB = {"balanced": "Balanced", "embb_intensive": "eMBB-int", "urllc_intensive": "URLLC-int", "mmtc_intensive": "mMTC-int"}
LAB2 = {"balanced": "Balanced", "embb_intensive": "eMBB\nint", "urllc_intensive": "URLLC\nint", "mmtc_intensive": "mMTC\nint"}
C_EMBB, C_URLLC, C_MMTC = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1b2430", "#5b6675", "#d9dee5"
CFG_COL = {"v1": "#7a8797", "v2": "#E69F00", "v3": "#0b3d91"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 20, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 1.6, "svg.fonttype": "path",
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def pair(run, a, b):
    return DATA[run]["pairs"][f"{a}->{b}"]


def gap_matrix(run):
    m = np.zeros((4, 4))
    sig = np.zeros((4, 4), bool)
    for i, a in enumerate(ENVS):
        for j, b in enumerate(ENVS):
            p = pair(run, a, b)["performance_gap"]
            m[i, j], sig[i, j] = p["gap"], p["significant"]
    return m, sig


# ---- 1. transfer-gap heatmap (v3) ---------------------------------------
def fig_heatmap(run="v3", name="gap_heatmap"):
    m, sig = gap_matrix(run)
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    norm = TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)
    ax.imshow(m, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(4):
        for j in range(4):
            if i == j:
                ax.text(j, i, "native\nref.", ha="center", va="center", fontsize=15, color=MUTED)
                continue
            t = f"{m[i, j]:+.3f}" + ("*" if sig[i, j] else "")
            ax.text(j, i, t, ha="center", va="center", fontsize=17, fontweight="bold",
                    color="white" if abs(m[i, j]) > 0.085 else INK)
    ax.set_xticks(range(4), [LAB2[e] for e in ENVS], fontsize=17)
    ax.set_yticks(range(4), [LAB[e] for e in ENVS], fontsize=17)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Evaluated on  B", fontsize=20, labelpad=12)
    ax.set_ylabel("Trained on  A", fontsize=20)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, name)


# ---- 2. signal vs noise ---------------------------------------------------
def fig_noise():
    cfgs = ["v1", "v2", "v3"]
    gap, noise = [], []
    for r in cfgs:
        g = [abs(pair(r, a, b)["performance_gap"]["gap"]) for a in ENVS for b in ENVS if a != b]
        n = [pair(r, e, e)["performance_gap"]["ssr_native_std"] for e in ENVS]
        gap.append(np.mean(g))
        noise.append(np.mean(n))
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    w = 0.36
    b1 = ax.bar(x - w / 2, gap, w, color="#0b3d91", label="Transfer gap (mean |gap|)")
    b2 = ax.bar(x + w / 2, noise, w, color="#E69F00", label="Seed-to-seed SD (same env.)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=18, fontweight="bold")
    for i in range(3):
        ax.text(x[i], -0.0155, f"ratio {gap[i] / noise[i]:.2f}", ha="center", va="top", fontsize=18, color=MUTED)
    ax.set_xticks(x, ["v1\n(3 seeds)", "v2\n(5 seeds)", "v3\n(5 seeds)"], fontsize=18)
    ax.tick_params(axis="x", pad=44)
    ax.set_ylabel("SSR", fontsize=20)
    ax.set_ylim(0, 0.135)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=15, loc="upper center", ncol=1)
    save(fig, "noise_vs_gap")
    return gap, noise


# ---- 3. distance vs gap scatter -------------------------------------------
def fig_scatter():
    fig, ax = plt.subplots(figsize=(6.9, 2.8))
    stats = {}
    for r in ["v1", "v2", "v3"]:
        xs, ys = [], []
        for a in ENVS:
            for b in ENVS:
                if a == b:
                    continue
                p = pair(r, a, b)
                xs.append(p["distribution_distance"]["wasserstein_mean"])
                ys.append(p["performance_gap"]["gap"])
        xs, ys = np.array(xs), np.array(ys)
        c = np.corrcoef(xs, ys)[0, 1]
        stats[r] = c
        ax.scatter(xs, ys, s=130, color=CFG_COL[r], alpha=0.85, edgecolor="white", linewidth=1.2,
                   label=f"{r}: r={c:+.2f}")
    ax.axhline(0, color=MUTED, lw=1.4, ls="--")
    ax.set_xlabel("1-Wasserstein distance (D_AB vs D_BB)", fontsize=17)
    ax.set_ylabel("Gap (SSR)", fontsize=19)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.14, 0.34)
    ax.legend(frameon=False, fontsize=16, loc="upper right", ncol=3, columnspacing=0.8, handletextpad=0.1)
    save(fig, "distance_vs_gap")
    return stats


# ---- 4. policy comparison vs heuristics -----------------------------------
def fig_policies():
    pol = [("Fixed 50/30/20", "fixed_50_30_20", "#a9b3c0"), ("Random", "random", "#6b7686"),
           ("Demand-prop.", "demand_proportional", "#E69F00")]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.arange(4)
    w = 0.2
    vals = []
    for k, (lbl, key, col) in enumerate(pol):
        v = [BASE[e][key]["ssr"] for e in ENVS]
        ax.bar(x + (k - 1.5) * w, v, w, color=col, label=lbl)
    dq = [pair("v3", e, e)["performance_gap"]["ssr_native"] for e in ENVS]
    sd = [pair("v3", e, e)["performance_gap"]["ssr_native_std"] for e in ENVS]
    bars = ax.bar(x + 1.5 * w, dq, w, color="#0b3d91", label="DQN (trained natively, v3)", yerr=sd,
                  error_kw=dict(ecolor=INK, capsize=5, lw=2))
    ax.set_xticks(x, [LAB2[e] for e in ENVS], fontsize=16)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Mean SSR", fontsize=20)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=16, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.38),
              columnspacing=1.0, handlelength=1.2)
    save(fig, "policies")


# ---- 5. training curves (v3) ----------------------------------------------
def fig_training():
    d = DATA["v3"]["train_rewards"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 2.9), sharey=True, sharex=True)
    axes = axes.ravel()
    k = 25
    for ax, e in zip(axes, ENVS):
        arr = np.array(d[e])  # seeds x episodes
        sm = np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in arr])
        ep = np.arange(sm.shape[1]) + k
        ax.fill_between(ep, sm.min(0), sm.max(0), color="#0b3d91", alpha=0.18, lw=0)
        ax.plot(ep, sm.mean(0), color="#0b3d91", lw=3)
        ax.set_title(LAB[e], fontsize=19, fontweight="bold")
        ax.set_xticks([0, 200, 400, 600])
        ax.tick_params(labelsize=15)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Reward", fontsize=17)
    axes[2].set_ylabel("Reward", fontsize=17)
    axes[2].set_xlabel("Episode", fontsize=17)
    axes[3].set_xlabel("Episode", fontsize=17)
    fig.subplots_adjust(wspace=0.08, hspace=0.4)
    save(fig, "training")


# ---- 6. allocation behaviour ----------------------------------------------
def fig_alloc():
    offered = {"balanced": (16, 16, 16), "embb_intensive": (24, 12, 12),
               "urllc_intensive": (12, 24, 12), "mmtc_intensive": (12, 12, 24)}
    fig, ax = plt.subplots(figsize=(6.9, 5.2))
    x = np.arange(4)
    w = 0.34
    cols = [C_EMBB, C_URLLC, C_MMTC]
    names = ["eMBB", "URLLC", "mMTC"]
    for xi, e in zip(x, ENVS):
        off = np.array(offered[e]) / 48
        mf = pair("v3", e, e)["performance_gap"]  # noqa: F841 (ensures key exists)
        agg = DATA["v3"]["agg"][f"{e}->{e}"]["distribution"]["mean_fraction"]
        pol = np.array([agg["eMBB"], agg["URLLC"], agg["mMTC"]])
        for bar_x, vals, hatch in [(xi - w / 2 - 0.02, off, "///"), (xi + w / 2 + 0.02, pol, None)]:
            bottom = 0
            for v, c in zip(vals, cols):
                ax.bar(bar_x, v, w, bottom=bottom, color=c, hatch=hatch, edgecolor="white", linewidth=1.5,
                       alpha=0.55 if hatch else 1)
                if v > 0.1:
                    ax.text(bar_x, bottom + v / 2, f"{v * 100:.0f}", ha="center", va="center", fontsize=15,
                            color="white", fontweight="bold")
                bottom += v
    ax.set_xticks(x, [LAB[e] for e in ENVS], fontsize=18)
    ax.set_ylabel("Share of total", fontsize=19)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
    save(fig, "allocation")


# ---- 7. source ranking ----------------------------------------------------
def fig_source():
    cfgs = ["v1", "v2", "v3"]
    x = np.arange(4)
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.9, 3.3))
    for k, r in enumerate(cfgs):
        vals = []
        for a in ENVS:
            vals.append(np.mean([pair(r, a, b)["performance_gap"]["gap"] for b in ENVS if b != a]))
        ax.bar(x + (k - 1) * w, vals, w, color=CFG_COL[r], label=r)
    ax.axhline(0, color=INK, lw=1.6)
    ax.set_xticks(x, [LAB2[e] for e in ENVS], fontsize=16)
    ax.set_ylabel("Mean outbound gap", fontsize=17)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=16, ncol=1, loc="upper right")
    save(fig, "source_rank")


if __name__ == "__main__":
    fig_heatmap()
    g, n = fig_noise()
    print("noise:", [round(v, 4) for v in g], [round(v, 4) for v in n], [round(a / b, 2) for a, b in zip(g, n)])
    print("corr:", fig_scatter())
    fig_policies()
    fig_training()
    fig_alloc()
    fig_source()
    for r in ["v1", "v3"]:
        m, _ = gap_matrix(r)
        print(r, np.round(m, 3).tolist())
    print("done")
