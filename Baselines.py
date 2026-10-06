"""Non-learning reference policies, evaluated under the same protocol as
Transfer.py (fixed per-environment seed, 20 episodes x 500 steps), so the
report's baseline table (Section 5.1) is reproducible from what is actually
in this repository rather than from an ad hoc, uncommitted script.

Run directly:  python Baselines.py
"""

import json
import os

import numpy as np

from Simulator import Simulator
from Environments import ENVIRONMENTS
from Allocators.FixedAllocator import FixedAllocator
from Allocators.RandomAllocator import RandomAllocator
from Allocators.DemandProportionalAllocator import DemandProportionalAllocator

ENV_KEYS = ["balanced", "embb_intensive", "urllc_intensive", "mmtc_intensive"]
EVAL_SEED_BASE = 9000  # matches Transfer.py, so baselines face the same traffic
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "baselines.json")

POLICIES = {
    "random": RandomAllocator,
    "fixed_50_30_20": FixedAllocator,
    "demand_proportional": DemandProportionalAllocator,
}


def evaluate(policy_cls, env_key, episodes=20, steps=500):
    env = ENVIRONMENTS[env_key]
    seed = EVAL_SEED_BASE + ENV_KEYS.index(env_key)
    np.random.seed(seed)

    ssrs = []
    for _ in range(episodes):
        policy = policy_cls(env["total_prb"])
        sim = Simulator(allocator=policy, env=env, steps=steps)
        _, _, _, _, _, ssr, _ = sim.run()
        ssrs.append(float(ssr.mean()))

    return {"ssr": float(np.mean(ssrs)), "ssr_std": float(np.std(ssrs, ddof=1))}


if __name__ == "__main__":
    results = {}
    for env_key in ENV_KEYS:
        results[env_key] = {}
        for name, cls in POLICIES.items():
            res = evaluate(cls, env_key)
            results[env_key][name] = res
            print(f"{env_key:<16} {name:<20} SSR={res['ssr']:.3f} (sd={res['ssr_std']:.3f})")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {RESULTS_PATH}")
