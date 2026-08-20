"""Balanced environment — reference scenario, load split evenly across slices.

Offered load: eMBB 16.0 / URLLC 16.0 / mMTC 16.0 = 48.0 PRB/step (33/33/33).

All four environments are load-matched at 48.0 PRB/step total against a 50 PRB
budget (96% offered utilisation) so that cross-environment gap numbers are
directly comparable — only the *mix* differs between them. Task sizes are identical everywhere; only arrival parameters change.
SLA thresholds are global (Environments/__init__.py: SLA).

Only the numbers below change between environments; each slice's arrival
algorithm is fixed for the whole study (see Environments/__init__.py).
"""

DESCRIPTION = "Balanced: load split evenly, 33/33/33"

TOTAL_PRB = 50

# eMBB — arrivals: ON/OFF Markov-modulated Poisson (bursty)
#   on_prob   P(stay ON | currently ON)
#   off_prob  P(switch to ON | currently OFF)
#   on_rate   Poisson lambda while ON
#   off_rate  Poisson lambda while OFF
#   duty cycle = off_prob / (off_prob + (1 - on_prob)) = 1/6
#   mean rate = 0.25 * on_rate  (with off_rate = 0.1 * on_rate)
EMBB = {
    "size_range": (20, 50),   # task size in PRBs, inclusive; mean 35

    "on_prob":    0.5,
    "off_prob":   0.1,
    "on_rate":    1.8286,         # mean 0.4572 tasks/step -> 16.0 PRB/step
    "off_rate":   0.18286,
}

# URLLC — arrivals: periodic (deterministic)
#   period      steps between arrival batches
#   batch_size  tasks per batch
#   mean rate = batch_size / period
URLLC = {
    "size_range": (2, 8),     # mean 5

    "period":     5,
    "batch_size": 16,             # mean 3.20 tasks/step -> 16.0 PRB/step
}

# mMTC — arrivals: Poisson
#   rate  lambda, tasks per step
MMTC = {
    "size_range": (1, 3),     # mean 2

    "rate":       8.0,            # mean 8.00 tasks/step -> 16.0 PRB/step
}
