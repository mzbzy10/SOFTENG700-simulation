import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Allocators"))

from FixedAllocator import FixedAllocator
from Simulator import Simulator

def run_simulator(
    allocator_cls=FixedAllocator,
    steps=100,
    total_prb=10,
    arrival_rate=5,
    job_size_mean=10
):
    allocator = allocator_cls(total_prb=total_prb)

    sim = Simulator(
        allocator=allocator,
        steps=steps,
        arrival_rate=arrival_rate,
        job_size_mean=job_size_mean
    )

    d, s, q, a, r, dm, aw = sim.run()

    sim.visualize()
run_simulator()