"""Environment registry: fixed structure here, tunable numbers in the env files.

Add or edit scenarios by editing balanced.py / embb_intensive.py /
urllc_intensive.py / mmtc_intensive.py. Nothing in this file is meant to be
tuned per scenario.
"""

import numpy as np

from . import balanced, embb_intensive, urllc_intensive, mmtc_intensive

SLICE_NAMES = ["eMBB", "URLLC", "mMTC"]

# Arrival algorithm per slice — FIXED for the whole study. Environments vary the
# parameters of these processes, never the choice of process.
ARRIVAL_MODELS = {
    "eMBB":  "onoff",
    "URLLC": "periodic",
    "mMTC":  "poisson",
}

# Parameters each algorithm requires. Used to validate the env files, so a typo
# in a scenario fails loudly at import instead of silently changing the traffic.
ARRIVAL_PARAMS = {
    "onoff":    {"on_prob", "off_prob", "on_rate", "off_rate"},
    "periodic": {"period", "batch_size"},
    "poisson":  {"rate"},
}

# Per-slice SLA parameters, present in every env file alongside the arrival ones.
SLA_PARAMS = {"size_range", "deadline"}


# --------------------------------------------------------------------------
# Normalization caps — deliberately global, NOT per-environment.
#
# get_state() divides the raw observation by these, and the reward divides the
# queue term by NORM_MAX_QUEUE. If they were derived per-environment (as they
# used to be, from that environment's own arrival rate), a policy trained on
# env A and evaluated on env B would see the *same physical demand* scaled by
# two different constants. Any measured transfer gap would then be partly a
# scaling artifact rather than a real policy mismatch, which would invalidate
# the cross-environment comparison. Keep these fixed across all environments.
#
# Values below reproduce what the balanced config computed previously:
#   demand cap = largest task size x highest arrival rate, per slice
#   queue cap  = 3 (mMTC rate) x 20
# NOTE: the demand caps clip fairly aggressively, since queued backlog can far
# exceed one step's worth of arrivals. Worth revisiting once the environments
# are finalized.
# --------------------------------------------------------------------------
NORM_MAX_DEMAND   = np.array([150.0, 8.0, 9.0])
NORM_MAX_QUEUE    = np.array([60.0, 60.0, 60.0])
NORM_MAX_DEADLINE = np.array([80.0, 10.0, 100.0])


# attribute name each slice uses inside an environment module
_SLICE_ATTR = {"eMBB": "EMBB", "URLLC": "URLLC", "mMTC": "MMTC"}


def _build(key, module):
    """Turn an environment module into the dict the Simulator consumes."""
    slices, arrivals = {}, {}

    for name in SLICE_NAMES:
        params = getattr(module, _SLICE_ATTR[name])
        model = ARRIVAL_MODELS[name]

        expected = SLA_PARAMS | ARRIVAL_PARAMS[model]
        actual = set(params)
        if actual != expected:
            missing, extra = expected - actual, actual - expected
            raise ValueError(
                f"{key}.{_SLICE_ATTR[name]} ({model} arrivals): "
                f"missing={sorted(missing)} unexpected={sorted(extra)}"
            )

        slices[name] = {k: params[k] for k in SLA_PARAMS}
        arrivals[name] = {k: params[k] for k in ARRIVAL_PARAMS[model]}

    return {
        "key":         key,
        "description": module.DESCRIPTION,
        "total_prb":   module.TOTAL_PRB,
        "slices":      slices,
        "arrivals":    arrivals,
    }


ENVIRONMENTS = {
    key: _build(key, module)
    for key, module in [
        ("balanced",        balanced),
        ("embb_intensive",  embb_intensive),
        ("urllc_intensive", urllc_intensive),
        ("mmtc_intensive",  mmtc_intensive),
    ]
}


def mean_arrival_rate(slice_name, arrivals):
    """Mean tasks per step for one slice, under its fixed arrival algorithm."""
    cfg = arrivals[slice_name]
    model = ARRIVAL_MODELS[slice_name]

    if model == "poisson":
        return cfg["rate"]

    if model == "onoff":
        duty = cfg["off_prob"] / (cfg["off_prob"] + (1 - cfg["on_prob"]))
        return duty * cfg["on_rate"] + (1 - duty) * cfg["off_rate"]

    if model == "periodic":
        return cfg["batch_size"] / cfg["period"]

    raise ValueError(f"unknown arrival model: {model}")


def offered_load(env):
    """Mean PRB-demand arriving per step, per slice.

    Use this to check whether environments are load-matched: if you want the gap
    numbers to be comparable across scenarios, keep the total roughly constant
    and vary only the mix.
    """
    load = np.zeros(len(SLICE_NAMES))

    for i, name in enumerate(SLICE_NAMES):
        low, high = env["slices"][name]["size_range"]
        avg_size = (low + high) / 2
        load[i] = avg_size * mean_arrival_rate(name, env["arrivals"])

    return load
