import numpy as np


class DemandProportionalAllocator:
    """Splits the PRB budget in proportion to each slice's current queued
    backlog (remaining work across all its incomplete tasks). Falls back to an
    even split when nothing is queued.

    Integer PRBs are assigned by the largest-remainder method (floor each
    share, then hand out the leftover PRBs to the slices with the largest
    fractional remainder) rather than dumping the rounding remainder into one
    fixed slice, so the bias introduced by rounding does not systematically
    favour or penalise any one slice.
    """

    def __init__(self, total_prb=50):
        self.total_prb = total_prb

    def get_allocation(self, requests, state=None):
        demand = np.zeros(3)
        for task in requests:
            demand[_SLICE_INDEX[task.slice_type]] += task.remaining

        if demand.sum() <= 0:
            base = self.total_prb // 3
            alloc = np.full(3, base, dtype=int)
            alloc[0] += self.total_prb - alloc.sum()
            return alloc

        share = demand / demand.sum() * self.total_prb
        alloc = np.floor(share).astype(int)
        remainder = self.total_prb - alloc.sum()

        if remainder > 0:
            order = np.argsort(-(share - alloc))
            for i in order[:remainder]:
                alloc[i] += 1

        return alloc


_SLICE_INDEX = {"eMBB": 0, "URLLC": 1, "mMTC": 2}
