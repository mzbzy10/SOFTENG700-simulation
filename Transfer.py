"""Cross-environment transfer experiment (CLAUDE.md step 4).

For every ordered pair (A, B) of environments this measures:

  Performance gap on B    SSR(policy trained on B, evaluated on B)
                        - SSR(policy trained on A, evaluated on B)

  Distributional distance  between D_AB (allocations the A-policy makes on B)
                           and D_BB (allocations the B-policy makes on B)

Everything runs from one entry point: `run_transfer_matrix()` trains N_SEEDS
policies per environment, then evaluates every policy on every environment on
the *same* traffic realisation (fixed evaluation seed per environment), so a
difference between D_AB and D_BB is a difference in policy, not in the traffic
it saw. Reported numbers are means over seeds with the across-seed standard
deviation, because a single DQN run is not a reproducible measurement.

Definitions used throughout:

  SSR      mean over steps and over the three slices of the per-slice SLA
           satisfaction ratio (Simulator.get_ssr). Scalar in [0, 1]. This is the
           same quantity the reward optimises, minus the queue shaping term.

  D_XY     the allocation distribution of policy X on environment Y, recorded in
           two forms because the two distances need different supports:
             - `action_probs`: categorical distribution over the DQN's discrete
               action set (the PRB splits). Used for KL / JS / TV. Valid only
               because every environment shares total_prb, hence one action set.
             - `fractions`: the raw (T, 3) samples of alloc / total_prb. Used for
               the per-slice 1-Wasserstein distance, which needs a ground metric
               and would be meaningless over unordered action indices.

           Caveat worth carrying into the write-up: when policies are close to
           deterministic their supports barely overlap, and then KL saturates at
           the smoothing floor while JS pins at ln 2 and TV at 1. Those three say
           only "different modal action". Wasserstein stays graded and should be
           the headline distance; KL is reported because CLAUDE.md asks for it.

Run directly:  python Transfer.py
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Allocators"))

from DQNAllocator import DQNAllocator
from Simulator import Simulator
from Environments import ENVIRONMENTS, SLICE_NAMES

ENV_KEYS = ["balanced", "embb_intensive", "urllc_intensive", "mmtc_intensive"]

_ROOT = os.path.dirname(__file__)

# Every run writes into results/<tag>/ and models/<tag>/ so a new experiment
# cannot overwrite an earlier one. Changing the simulator (SLA definitions,
# reward, agent) makes old and new numbers incomparable, so the artifacts have
# to be kept side by side rather than replaced in place.
RUN_TAG = "v2_ddqn_servedratio"
RESULTS_DIR = os.path.join(_ROOT, "results", RUN_TAG)
MODELS_DIR = os.path.join(_ROOT, "models", RUN_TAG)


def set_run_tag(tag):
    """Point the results and models directories at a named run group."""
    global RUN_TAG, RESULTS_DIR, MODELS_DIR
    RUN_TAG = tag
    RESULTS_DIR = os.path.join(_ROOT, "results", tag)
    MODELS_DIR = os.path.join(_ROOT, "models", tag)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    return RUN_TAG

# Independent training runs per environment. Three is the minimum that gives a
# usable spread; a single DQN run collapses often enough that one seed cannot
# distinguish a real transfer gap from a bad initialisation.
N_SEEDS = 3

# Base seeds. Training seeds differ per (environment, run) so the policies are
# independent; the evaluation seed depends on the *environment only*, so every
# policy evaluated on B faces the identical arrival sequence.
TRAIN_SEED_BASE = 1000
EVAL_SEED_BASE = 9000

# Fraction of training during which epsilon is still above its floor. The
# DQNAllocator default (0.9995 per gradient step) hits epsilon_min after ~6k
# steps, i.e. 12 of 150 episodes — the agent stops exploring before its own
# Q-values are worth trusting, which is how policies end up locked onto two or
# three actions. The decay is instead derived from the actual training length so
# exploration spans most of the run.
EXPLORE_FRACTION = 0.6


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_decay_for(total_steps, epsilon_min=0.05, epsilon_start=1.0,
                      explore_fraction=EXPLORE_FRACTION):
    """Per-step multiplicative decay that reaches epsilon_min at the given point.

    Solves  epsilon_start * d^n = epsilon_min  for n = explore_fraction * total_steps.
    """
    n = max(1.0, explore_fraction * total_steps)
    return float(np.exp(np.log(epsilon_min / epsilon_start) / n))


class FrozenPolicy:
    """Greedy, non-learning view of a trained DQNAllocator.

    The Simulator drives learning through `store` / `train_step` /
    `update_target`, which it discovers with hasattr. This wrapper deliberately
    exposes none of them, so evaluation cannot mutate the policy, and it always
    takes the argmax action (no epsilon-greedy exploration) so D_XY reflects the
    learned policy rather than its exploration noise.
    """

    def __init__(self, dqn):
        self.dqn = dqn
        self.total_prb = dqn.total_prb
        self.actions = dqn.actions
        self.last_action_idx = None
        self.action_idx_hist = []

    def get_allocation(self, requests, state=None):
        if state is None:
            idx = 0
        else:
            with torch.no_grad():
                q = self.dqn.q_net(torch.FloatTensor(np.asarray(state, dtype=np.float32)).unsqueeze(0))
            idx = int(q.argmax().item())

        self.last_action_idx = idx
        self.action_idx_hist.append(idx)
        return self.actions[idx].copy()


# --------------------------------------------------------------------------
# training and evaluation
# --------------------------------------------------------------------------

def train_policy(env_key, episodes=150, steps=500, seed=None, verbose=True,
                 epsilon_decay=None):
    """Train a fresh DQN policy on one environment.

    Returns the allocator, with the per-episode reward history attached as
    `.train_rewards` so convergence can be inspected afterwards.
    """
    env = ENVIRONMENTS[env_key]
    if seed is None:
        seed = TRAIN_SEED_BASE + ENV_KEYS.index(env_key)
    set_seed(seed)

    if epsilon_decay is None:
        epsilon_decay = epsilon_decay_for(episodes * steps)

    allocator = DQNAllocator(total_prb=env["total_prb"], epsilon_decay=epsilon_decay)
    sim = Simulator(allocator=allocator, env=env, steps=steps)

    episode_rewards = []
    for ep in range(1, episodes + 1):
        if ep > 1:
            sim.reset()
        *_, r, _, _ = sim.run()
        episode_rewards.append(float(r.sum()))

        if verbose and (ep % 25 == 0 or ep == episodes):
            recent = np.mean(episode_rewards[-25:])
            print(f"    ep {ep:>4}/{episodes}  reward(last25)={recent:8.2f}  eps={allocator.epsilon:.3f}")

    allocator.train_rewards = episode_rewards
    return allocator


def evaluate_policy(allocator, env_key, episodes=20, steps=500, seed=None):
    """Evaluate a trained policy on one environment, greedily and without learning.

    Returns a dict holding the scalar SSR, its per-episode spread, the per-slice
    breakdown, and the allocation distribution D (see module docstring).
    """
    env = ENVIRONMENTS[env_key]
    if seed is None:
        seed = EVAL_SEED_BASE + ENV_KEYS.index(env_key)
    set_seed(seed)

    policy = FrozenPolicy(allocator)
    sim = Simulator(allocator=policy, env=env, steps=steps)

    ep_ssr, ep_reward = [], []
    ssr_slices, alloc_samples = [], []

    for ep in range(episodes):
        if ep > 0:
            sim.reset()
        _, _, _, alloc, reward, ssr, _ = sim.run()

        ep_ssr.append(float(ssr.mean()))
        ep_reward.append(float(reward.sum()))
        ssr_slices.append(ssr)
        alloc_samples.append(alloc)

    ssr_all = np.concatenate(ssr_slices, axis=0)          # (episodes*steps, 3)
    alloc_all = np.concatenate(alloc_samples, axis=0)      # (episodes*steps, 3)

    n_actions = len(allocator.actions)
    counts = np.bincount(np.array(policy.action_idx_hist), minlength=n_actions)

    return {
        "env": env_key,
        "ssr": float(ssr_all.mean()),
        "ssr_std_per_episode": float(np.std(ep_ssr, ddof=1)) if len(ep_ssr) > 1 else 0.0,
        "ssr_per_slice": {s: float(ssr_all[:, i].mean()) for i, s in enumerate(SLICE_NAMES)},
        "reward_per_episode": float(np.mean(ep_reward)),
        "episodes": episodes,
        "steps": steps,
        "distribution": make_distribution(counts, alloc_all, allocator.total_prb),
    }


def make_distribution(counts, alloc, total_prb):
    """Package raw action counts and allocations into a distribution dict D."""
    counts = np.asarray(counts, dtype=float)
    fractions = np.asarray(alloc, dtype=float) / float(total_prb)
    return {
        "action_counts": counts,
        "action_probs": counts / counts.sum(),
        "fractions": fractions,
        "mean_fraction": {s: float(fractions[:, i].mean()) for i, s in enumerate(SLICE_NAMES)},
        "actions_used": int((counts > 0).sum()),
    }


def pool_distributions(dists):
    """Merge the D's of several seeds into one, by pooling their raw samples.

    Pooling rather than averaging the probability vectors: the pooled object is
    still an empirical distribution over real (state -> action) decisions, so the
    same distance functions apply unchanged.
    """
    counts = np.sum([d["action_counts"] for d in dists], axis=0)
    fractions = np.concatenate([d["fractions"] for d in dists], axis=0)
    total_prb = 1.0   # fractions are already normalised
    pooled = make_distribution(counts, fractions, total_prb)
    return pooled


# --------------------------------------------------------------------------
# distributional distances
# --------------------------------------------------------------------------

def kl_divergence(p, q, eps=1e-6):
    """KL(p || q) in nats, over a shared categorical support.

    Both are Laplace-smoothed by `eps` first: a greedy DQN puts exactly zero mass
    on most actions, and an unsmoothed zero in q makes KL infinite, which carries
    no information about *how* far apart the policies are. The flip side is that
    when supports are disjoint the value saturates near ln(1/eps) and stops being
    a distance — check `actions_used` before reading much into a large KL.
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q, eps=1e-6):
    """Jensen-Shannon divergence — symmetric, bounded by ln 2, smoothing-insensitive."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def total_variation(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(0.5 * np.abs(p - q).sum())


def wasserstein_1d(x, y, n_quantiles=1000):
    """1-Wasserstein (earth mover's) distance between two 1-D sample sets.

    Computed from the quantile functions, |F^-1_x - F^-1_y| integrated over
    [0,1], so the two samples need not be the same length. Implemented here
    rather than pulled from scipy to keep the dependency set at numpy/torch.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def allocation_wasserstein(dist_p, dist_q):
    """Per-slice W1 between two allocation distributions, on PRB *fractions*.

    Fractions rather than raw PRBs so the number is scale-free and comparable
    across any future change of budget. Returns per-slice values plus their mean.
    """
    fx = np.asarray(dist_p["fractions"])
    fy = np.asarray(dist_q["fractions"])
    per_slice = {s: wasserstein_1d(fx[:, i], fy[:, i]) for i, s in enumerate(SLICE_NAMES)}
    return {"per_slice": per_slice, "mean": float(np.mean(list(per_slice.values())))}


def distribution_distance(dist_transfer, dist_native):
    """All distances between D_AB (transferred) and D_BB (native), in one dict."""
    p = dist_transfer["action_probs"]
    q = dist_native["action_probs"]
    w = allocation_wasserstein(dist_transfer, dist_native)

    return {
        "kl": kl_divergence(p, q),
        "kl_reverse": kl_divergence(q, p),
        "js": js_divergence(p, q),
        "total_variation": total_variation(p, q),
        "wasserstein_mean": w["mean"],
        "wasserstein_per_slice": w["per_slice"],
        "mean_fraction_shift": {
            s: dist_transfer["mean_fraction"][s] - dist_native["mean_fraction"][s]
            for s in SLICE_NAMES
        },
    }


def performance_gap(ssr_transfer, ssr_native):
    """Native SSR minus transferred SSR, from per-seed score lists.

    Positive => the native policy is better, i.e. transfer costs something.
    The standard error combines both sides, so a gap smaller than roughly twice
    it is indistinguishable from seed noise and must not be read as a transfer
    effect.
    """
    t = np.asarray(ssr_transfer, dtype=float)
    n = np.asarray(ssr_native, dtype=float)
    gap = float(n.mean() - t.mean())

    def _sem(x):
        return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    sem = float(np.hypot(_sem(t), _sem(n)))
    return {
        "ssr_native": float(n.mean()),
        "ssr_native_std": float(np.std(n, ddof=1)) if len(n) > 1 else 0.0,
        "ssr_transferred": float(t.mean()),
        "ssr_transferred_std": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
        "gap": gap,
        "gap_sem": sem,
        "significant": bool(abs(gap) > 2 * sem) if sem > 0 else False,
        "relative_gap": gap / n.mean() if n.mean() > 0 else float("nan"),
        "ssr_native_per_seed": n.tolist(),
        "ssr_transferred_per_seed": t.tolist(),
    }


# --------------------------------------------------------------------------
# the whole experiment, one call
# --------------------------------------------------------------------------

def run_transfer_matrix(
    env_keys=None,
    n_seeds=N_SEEDS,
    train_episodes=150,
    eval_episodes=20,
    steps=500,
    save_models=True,
    reuse_models=False,
    verbose=True,
):
    """Train n_seeds policies per environment, cross-evaluate all pairs, compute gaps.

    Returns a dict with:
      policies[env]                       -> list of trained allocators
      train_rewards[env]                  -> list of per-episode reward curves
      evaluations[(train_env, s, eval_env)] -> per-seed evaluation dict
      agg[(train_env, eval_env)]          -> seed-aggregated SSR + pooled D
      pairs[(A, B)]                       -> {performance_gap, distribution_distance}
    """
    env_keys = env_keys or ENV_KEYS
    os.makedirs(MODELS_DIR, exist_ok=True)

    decay = epsilon_decay_for(train_episodes * steps)
    if verbose:
        print(f"epsilon decay {decay:.6f} per gradient step "
              f"(floor reached at ~{EXPLORE_FRACTION:.0%} of {train_episodes * steps} steps)")

    # --- 1. train n_seeds policies per environment ------------------------
    policies = {k: [] for k in env_keys}
    train_rewards = {k: [] for k in env_keys}

    for key in env_keys:
        for s in range(n_seeds):
            path = os.path.join(MODELS_DIR, f"policy_{key}_s{s}.pt")
            rpath = path.replace(".pt", "_rewards.json")

            if reuse_models and os.path.exists(path):
                if verbose:
                    print(f"\nReusing cached policy {key} seed {s}")
                allocator = DQNAllocator(total_prb=ENVIRONMENTS[key]["total_prb"])
                allocator.load(path, inference=True)
                allocator.train_rewards = (
                    json.load(open(rpath)) if os.path.exists(rpath) else []
                )
            else:
                if verbose:
                    print(f"\nTraining {key} seed {s} ({train_episodes} x {steps} steps)")
                allocator = train_policy(
                    key, episodes=train_episodes, steps=steps,
                    seed=TRAIN_SEED_BASE + 100 * ENV_KEYS.index(key) + s,
                    verbose=verbose, epsilon_decay=decay,
                )
                if save_models:
                    allocator.save(path)
                    with open(rpath, "w") as f:
                        json.dump(allocator.train_rewards, f)

            policies[key].append(allocator)
            train_rewards[key].append(allocator.train_rewards)

    # --- 2. evaluate every policy on every environment --------------------
    evaluations = {}
    for train_env in env_keys:
        for s, allocator in enumerate(policies[train_env]):
            for eval_env in env_keys:
                res = evaluate_policy(allocator, eval_env,
                                      episodes=eval_episodes, steps=steps)
                evaluations[(train_env, s, eval_env)] = res
                if verbose:
                    print(f"  policy[{train_env} s{s}] on {eval_env:<16} "
                          f"SSR={res['ssr']:.4f}  actions_used={res['distribution']['actions_used']}")

    # --- 3. aggregate over seeds ------------------------------------------
    agg = {}
    for a in env_keys:
        for b in env_keys:
            per_seed = [evaluations[(a, s, b)] for s in range(n_seeds)]
            agg[(a, b)] = {
                "ssr_per_seed": [e["ssr"] for e in per_seed],
                "ssr": float(np.mean([e["ssr"] for e in per_seed])),
                "ssr_std": float(np.std([e["ssr"] for e in per_seed], ddof=1)) if n_seeds > 1 else 0.0,
                "ssr_per_slice": {
                    sl: float(np.mean([e["ssr_per_slice"][sl] for e in per_seed]))
                    for sl in SLICE_NAMES
                },
                "actions_used_mean": float(np.mean([e["distribution"]["actions_used"] for e in per_seed])),
                "distribution": pool_distributions([e["distribution"] for e in per_seed]),
            }

    # --- 4. gaps and distances for every ordered pair ---------------------
    pairs = {}
    for a in env_keys:
        for b in env_keys:
            transferred = agg[(a, b)]   # D_AB
            native = agg[(b, b)]        # D_BB
            pairs[(a, b)] = {
                "train_env": a,
                "eval_env": b,
                "performance_gap": performance_gap(
                    transferred["ssr_per_seed"], native["ssr_per_seed"]
                ),
                "gap_per_slice": {
                    sl: native["ssr_per_slice"][sl] - transferred["ssr_per_slice"][sl]
                    for sl in SLICE_NAMES
                },
                "distribution_distance": distribution_distance(
                    transferred["distribution"], native["distribution"]
                ),
            }

    return {
        "env_keys": env_keys,
        "policies": policies,
        "train_rewards": train_rewards,
        "evaluations": evaluations,
        "agg": agg,
        "pairs": pairs,
        "config": {
            "run_tag": RUN_TAG,
            "n_seeds": n_seeds,
            "train_episodes": train_episodes,
            "eval_episodes": eval_episodes,
            "steps": steps,
            "epsilon_decay": decay,
            "explore_fraction": EXPLORE_FRACTION,
            "train_seed_base": TRAIN_SEED_BASE,
            "eval_seed_base": EVAL_SEED_BASE,
        },
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _matrix(results, value_fn, keys):
    return np.array([[value_fn(results["pairs"][(a, b)]) for b in keys] for a in keys])


def print_report(results):
    keys = results["env_keys"]
    short = {k: k.split("_")[0][:8] for k in keys}
    head = "".join(f"{short[k]:>14}" for k in keys)

    print("\n" + "=" * 78)
    print("SSR  mean +/- sd over seeds   (rows = trained on, cols = evaluated on)")
    print(f"{'':>16}{head}")
    for a in keys:
        row = ""
        for b in keys:
            g = results["agg"][(a, b)]
            row += f"{g['ssr']:.3f}+-{g['ssr_std']:.3f}".rjust(14)
        print(f"{short[a]:>16}{row}")

    print("\nPerformance gap on B  (native SSR - transferred SSR; * = |gap| > 2 SEM)")
    print(f"{'':>16}{head}")
    for a in keys:
        row = ""
        for b in keys:
            pg = results["pairs"][(a, b)]["performance_gap"]
            mark = "*" if pg["significant"] else " "
            row += f"{pg['gap']:+.3f}{mark}".rjust(14)
        print(f"{short[a]:>16}{row}")

    tables = [
        ("KL( D_AB || D_BB )  nats, over the discrete action set",
         lambda p: p["distribution_distance"]["kl"]),
        ("Jensen-Shannon  (max = ln 2 = 0.693; at the max the supports are disjoint)",
         lambda p: p["distribution_distance"]["js"]),
        ("Wasserstein-1  (mean over slices, PRB fraction) -- headline distance",
         lambda p: p["distribution_distance"]["wasserstein_mean"]),
    ]
    for title, fn in tables:
        m = _matrix(results, fn, keys)
        print(f"\n{title}")
        print(f"{'':>16}{head}")
        for i, a in enumerate(keys):
            print(f"{short[a]:>16}" + "".join(f"{v:14.4f}" for v in m[i]))

    print("\nNative policy behaviour (diagonal): mean PRB fraction and action-set usage")
    for b in keys:
        g = results["agg"][(b, b)]
        mf = g["distribution"]["mean_fraction"]
        print(f"  {b:<18}" + "  ".join(f"{s}={mf[s]:.3f}" for s in SLICE_NAMES)
              + f"   actions_used(mean over seeds)={g['actions_used_mean']:.1f}")

    # off-diagonal gap vs distance, the actual correlation of interest
    gaps, kls, jss, ws = [], [], [], []
    for a in keys:
        for b in keys:
            if a == b:
                continue
            p = results["pairs"][(a, b)]
            gaps.append(p["performance_gap"]["gap"])
            kls.append(p["distribution_distance"]["kl"])
            jss.append(p["distribution_distance"]["js"])
            ws.append(p["distribution_distance"]["wasserstein_mean"])

    def _corr(x, y):
        # undefined when either series is constant (e.g. a saturated distance)
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    if len(gaps) > 1:
        n_sig = sum(1 for a in keys for b in keys
                    if a != b and results["pairs"][(a, b)]["performance_gap"]["significant"])
        print("\nOff-diagonal summary (n = %d ordered pairs)" % len(gaps))
        print(f"  corr(gap, KL)          = {_corr(gaps, kls):.4f}")
        print(f"  corr(gap, JS)          = {_corr(gaps, jss):.4f}")
        print(f"  corr(gap, Wasserstein) = {_corr(gaps, ws):.4f}")
        print(f"  mean gap = {np.mean(gaps):+.4f}   min = {np.min(gaps):+.4f}   max = {np.max(gaps):+.4f}")
        print(f"  negative gaps (transfer beats native) = {sum(1 for g in gaps if g < 0)}/{len(gaps)}")
        print(f"  gaps exceeding 2 SEM = {n_sig}/{len(gaps)}")
    print("=" * 78)


def plot_training_curves(results, path=None, show=True):
    """One panel per environment: every seed's reward curve plus the seed mean.

    This is the convergence check — flat-from-the-start or wildly diverging seeds
    mean the transfer numbers are measuring initialisation luck, not transfer.
    """
    import matplotlib.pyplot as plt

    keys = results["env_keys"]
    curves = results["train_rewards"]
    n = len(keys)
    fig, axs = plt.subplots((n + 1) // 2, 2, figsize=(13, 4 * ((n + 1) // 2)))
    axs = np.atleast_1d(axs).flatten()

    for ax, key in zip(axs, keys):
        runs = [np.asarray(c, dtype=float) for c in curves[key] if len(c)]
        if not runs:
            ax.set_title(f"{key} — no reward history (cached model)")
            ax.axis("off")
            continue

        L = min(len(r) for r in runs)
        stacked = np.stack([r[:L] for r in runs])
        w = max(1, min(10, L // 5))
        kernel = np.ones(w) / w
        xs = np.arange(w - 1, L)

        for i, r in enumerate(stacked):
            ax.plot(xs, np.convolve(r, kernel, mode="valid"), alpha=0.45, label=f"seed {i}")
        ax.plot(xs, np.convolve(stacked.mean(axis=0), kernel, mode="valid"),
                color="black", lw=2, label="seed mean")

        ax.set_title(f"{key}  (episode reward, {w}-episode avg)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend(fontsize=8)

    for ax in axs[len(keys):]:
        ax.axis("off")

    plt.tight_layout()
    if path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=120)
    print(f"Training curves written to {path}")
    if show:
        plt.show()
    return path


def convergence_summary(results):
    """Per-environment numeric convergence check, printed alongside the plot.

    Compares the mean reward of the first and last fifth of training: a policy
    that never improved has nothing to transfer, and its row in the gap matrix
    should not be interpreted.
    """
    print("\nConvergence check (episode reward, first fifth -> last fifth)")
    for key in results["env_keys"]:
        for i, c in enumerate(results["train_rewards"][key]):
            if not c:
                print(f"  {key:<18} seed {i}: no history (cached)")
                continue
            c = np.asarray(c, dtype=float)
            k = max(1, len(c) // 5)
            first, last = c[:k].mean(), c[-k:].mean()
            tail_sd = float(np.std(c[-k:], ddof=1)) if k > 1 else 0.0
            print(f"  {key:<18} seed {i}: {first:8.1f} -> {last:8.1f} "
                  f"(delta {last - first:+8.1f}, tail sd {tail_sd:6.1f})")


def save_results(results, path=None):
    """Write everything except the torch models to JSON (raw samples dropped)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"transfer_{stamp}.json")

    def _clean_dist(d):
        d = dict(d)
        d.pop("fractions")                       # (N,3) float array, too large for JSON
        d["action_counts"] = np.asarray(d["action_counts"]).tolist()
        d["action_probs"] = np.asarray(d["action_probs"]).tolist()
        return d

    payload = {
        "config": results["config"],
        "env_keys": results["env_keys"],
        "train_rewards": results["train_rewards"],
        "evaluations": {},
        "agg": {},
        "pairs": {f"{a}->{b}": v for (a, b), v in results["pairs"].items()},
    }
    for (a, s, b), ev in results["evaluations"].items():
        ev = dict(ev)
        ev["distribution"] = _clean_dist(ev["distribution"])
        payload["evaluations"][f"{a}|s{s}->{b}"] = ev
    for (a, b), g in results["agg"].items():
        g = dict(g)
        g["distribution"] = _clean_dist(g["distribution"])
        payload["agg"][f"{a}->{b}"] = g

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-environment transfer experiment")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--envs", nargs="*", default=ENV_KEYS)
    ap.add_argument("--reuse", action="store_true", help="reuse cached models in models/ instead of retraining")
    ap.add_argument("--no-save", action="store_true", help="do not write models or JSON")
    ap.add_argument("--no-plot", action="store_true", help="skip the training-curve figure")
    ap.add_argument("--tag", default=RUN_TAG,
                    help="run group; artifacts go to results/<tag>/ and models/<tag>/")
    args = ap.parse_args()

    set_run_tag(args.tag)
    print(f"run tag: {args.tag}  ->  results/{args.tag}/  models/{args.tag}/")

    results = run_transfer_matrix(
        env_keys=args.envs,
        n_seeds=args.seeds,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        steps=args.steps,
        save_models=not args.no_save,
        reuse_models=args.reuse,
    )
    print_report(results)
    convergence_summary(results)
    if not args.no_save:
        save_results(results)
    if not args.no_plot:
        plot_training_curves(results, show=False)
