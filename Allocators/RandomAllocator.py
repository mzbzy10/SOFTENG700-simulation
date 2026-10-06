import numpy as np


class RandomAllocator:
    """Uniform over the same discretized action grid the DQN allocator uses
    (multiples of 5 PRBs, each slice >= 5), so the baseline is comparable to a
    DQN policy that has not learned anything yet.
    """

    def __init__(self, total_prb=50, step=5):
        self.total_prb = total_prb
        self.actions = np.array([
            [a, b, total_prb - a - b]
            for a in range(step, total_prb - step + 1, step)
            for b in range(step, total_prb - a, step)
            if total_prb - a - b >= step
        ])

    def get_allocation(self, requests, state=None):
        idx = np.random.randint(len(self.actions))
        return self.actions[idx].copy()
