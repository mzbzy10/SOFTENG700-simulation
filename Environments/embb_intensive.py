"""eMBB-intensive environment.

PLACEHOLDER — currently identical to balanced.py. Differentiate by raising
eMBB's share of the offered load (on_rate, duty cycle, or size_range), and
lowering the other two if you want the total load to stay matched.

Only the numbers below change between environments; each slice's arrival
algorithm is fixed for the whole study (see Environments/__init__.py).
"""

DESCRIPTION = "eMBB-intensive: PLACEHOLDER, currently identical to balanced"

TOTAL_PRB = 50

# eMBB — arrivals: ON/OFF Markov-modulated Poisson (bursty)
#   on_prob   P(stay ON | currently ON)
#   off_prob  P(switch to ON | currently OFF)
#   on_rate   Poisson lambda while ON
#   off_rate  Poisson lambda while OFF
#   duty cycle = off_prob / (off_prob + (1 - on_prob))
EMBB = {
    "size_range": (20, 50),   # task size in PRBs, inclusive
    "deadline":   80,         # steps

    "on_prob":    0.5,
    "off_prob":   0.1,
    "on_rate":    3,
    "off_rate":   0.3,
}

# URLLC — arrivals: periodic (deterministic)
#   period      steps between arrival batches
#   batch_size  tasks per batch
URLLC = {
    "size_range": (2, 8),
    "deadline":   10,

    "period":     4,
    "batch_size": 1,
}

# mMTC — arrivals: Poisson
#   rate  lambda, tasks per step
MMTC = {
    "size_range": (1, 3),
    "deadline":   100,

    "rate":       3,
}
