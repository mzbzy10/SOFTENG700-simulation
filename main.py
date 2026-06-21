import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Allocators"))

import matplotlib.pyplot as plt

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
    episodes=10,
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

    episode_rewards = []

    for ep in range(1, episodes + 1):
        if ep > 1:
            sim.reset()

        _, _, _, _, r, _, _ = sim.run()

        ep_reward = r.sum()
        episode_rewards.append(ep_reward)
        epsilon_str = f"  ε={allocator.epsilon:.3f}" if hasattr(allocator, 'epsilon') else ""
        print(f"Episode {ep}/{episodes}  total reward: {ep_reward:.2f}{epsilon_str}")

    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, episodes + 1), episode_rewards, marker='o', markersize=3)
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    plt.tight_layout()
    plt.show()

    sim.visualize()

    if isinstance(allocator, DQNAllocator):
        if input("\nSave model? (y/n): ").strip().lower() == 'y':
            path = input("Save path (default: dqn_model.pt): ").strip() or "dqn_model.pt"
            allocator.save(path)

if __name__ == "__main__":
    name, allocator_cls = select_model()
    print(f"\nRunning: {name}\n")
    run_simulator(allocator_cls=allocator_cls)
