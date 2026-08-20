"""mMTC-intensive environment — mMTC takes half the offered load.

Offered load: eMBB 12.0 / URLLC 12.0 / mMTC 24.0 = 48.0 PRB/step (25/25/50).

Load-matched to the other three environments at 48.0 PRB/step total; only the
mix differs. Task sizes are identical everywhere; only arrival parameters change.
SLA thresholds are global (Environments/__init__.py: SLA).

This is expected to be the most forgiving scenario: mMTC's load arrives as many
tiny tasks against a generous buffer SLA, so it absorbs deferral well.

Only the numbers below change between environments; each slice's arrival
algorithm is fixed for the whole study (see Environments/__init__.py).
"""

DESCRIPTION = "mMTC-intensive: mMTC takes half the load, 25/25/50"

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
    "on_rate":    1.3714,         # mean 0.3428 tasks/step -> 12.0 PRB/step
    "off_rate":   0.13714,
}

# URLLC — arrivals: periodic (deterministic)
#   period      steps between arrival batches
#   batch_size  tasks per batch
#   mean rate = batch_size / period
URLLC = {
    "size_range": (2, 8),     # mean 5

    "period":     5,
    "batch_size": 12,             # mean 2.40 tasks/step -> 12.0 PRB/step
}

# mMTC — arrivals: Poisson
#   rate  lambda, tasks per step
MMTC = {
    "size_range": (1, 3),     # mean 2

    "rate":       12.0,           # mean 12.00 tasks/step -> 24.0 PRB/step
}
