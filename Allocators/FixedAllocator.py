import numpy as np

class FixedAllocator:
    def __init__(self, total_prb=60):
        self.total_prb = total_prb
        self.ratio = np.array([0.5, 0.3, 0.2])

    def get_allocation(self, requests):
        alloc = (self.ratio * self.total_prb).astype(int)
        alloc[0] += self.total_prb - alloc.sum()
        return alloc