import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Allocators"))

from FixedAllocator import FixedAllocator
from DQNAllocator import DQNAllocator
from Simulator import Simulator

MODELS = [
    ("Fixed Allocator (baseline)", FixedAllocator),
    ("DQN Allocator (reinforcement learning)", DQNAllocator),
]

def select_model():
    print("\nAvailable models:")
    for i, (name, _) in enumerate(MODELS, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input("\nSelect a model (enter number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            return MODELS[int(choice) - 1]
        print(f"Invalid choice. Please enter a number between 1 and {len(MODELS)}.")

def run_simulator(
    allocator_cls=FixedAllocator,
    steps=500,
    total_prb=50,
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

if __name__ == "__main__":
    name, allocator_cls = select_model()
    print(f"\nRunning: {name}\n")
    run_simulator(allocator_cls=allocator_cls)
