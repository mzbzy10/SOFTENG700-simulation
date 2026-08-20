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

# Per-slice traffic parameters, present in every env file alongside the arrival
# ones. Note that `deadline` is NOT here: SLA thresholds are global (see SLA
# below), because they define what SSR *means* and must not vary per scenario.
SLA_PARAMS = {"size_range"}


# --------------------------------------------------------------------------
# SLA definitions — one KPI per slice type, global across all environments.
#
# The survey (Section IV-C) criticises formulations that "only consider one type
# of service with different threshold levels as different slices. Nevertheless,
# slices come with different KPI requirements and different impacting factors,
# so a single slice type cannot accommodate the heterogeneous set of services in
# 5G." Using one deadline metric with three thresholds is exactly that pattern,
# so each slice instead gets the KPI its service class is actually defined by,
# following Table I [6] ("SLA satisfaction in terms of average buffer length,
# data rate, and PRB usage"), [10] (throughput for eMBB, queuing delay for
# uRLLC) and Table III [12] ("minimum data rate and maximum delay").
#
#   eMBB   minimum data rate   — sustained PRBs/step served over a window
#   URLLC  maximum delay       — task waiting time, in steps
#   mMTC   maximum buffer      — queued task backlog
#
# These are contract terms, not traffic parameters: they MUST be identical in
# every environment. If env A promised a 5-step URLLC delay and env B promised
# 20, "SSR" would denote different things in each and a cross-environment
# transfer gap would be uninterpretable.
#
# Timebase: 1 step = 1 TTI = 1 ms, so max_delay 5 = 5 ms (3GPP URLLC user-plane
# latency targets are 1-10 ms).
#
# Thresholds are calibrated, not guessed: each was swept against both a
# demand-proportional (near-optimal) and the naive fixed policy across all four
# environments, and set where the good policy scores ~0.75-0.99 — off the 1.0
# ceiling so there is headroom to measure, but far from floored. Resulting
# per-slice SSR, greedy / fixed:
#
#             eMBB        URLLC        mMTC
#   balanced  0.97/0.95   0.69/0.13   0.80/0.04
#   embb      0.99/0.99   0.60/1.00   0.74/0.10
#   urllc     0.91/0.91   0.81/0.03   0.97/0.11
#   mmtc      0.85/0.87   0.99/1.00   0.95/0.02
#
# min_rate sits below every environment's eMBB offered load (12-24 PRB/step) so
# it acts as a floor rather than collapsing into "serve all demand".
# --------------------------------------------------------------------------
SLA = {
    "eMBB":  {"kpi": "min_rate",   "min_rate": 10.0, "window": 20},
    "URLLC": {"kpi": "max_delay",  "max_delay": 5},
    "mMTC":  {"kpi": "max_buffer", "max_buffer": 60},
}


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
# These are measured, not guessed: the p99 of each signal pooled over all four
# environments at the 48 PRB/step operating load, driven by a demand-proportional
# (near-optimal) policy. Sizing them to a good policy's operating range keeps the
# observation well resolved where the agent actually lives; a policy doing much
# worse clips at 1.0, which is itself the correct signal ("badly backlogged").
#
# Re-measure these if the offered load or the arrival mix changes materially.
# --------------------------------------------------------------------------
NORM_MAX_DEMAND = np.array([1200.0, 550.0, 450.0])
NORM_MAX_QUEUE  = np.array([  35.0, 105.0, 220.0])
NORM_MAX_WAIT   = np.array([  80.0,  20.0, 100.0])


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
