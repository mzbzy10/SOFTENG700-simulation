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
    episodes=50,
    steps=500,
    total_prb=50,
    arrival_rate=5
):
    allocator = allocator_cls(total_prb=total_prb)

    sim = Simulator(
        allocator=allocator,
        steps=steps,
        arrival_rate=arrival_rate
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

    sim.visualize(episode_rewards)

    if isinstance(allocator, DQNAllocator):
        if input("\nSave model? (y/n): ").strip().lower() == 'y':
            path = input("Save path (default: dqn_model.pt): ").strip() or "dqn_model.pt"
            allocator.save(path)

if __name__ == "__main__":
    name, allocator_cls = select_model()
    print(f"\nRunning: {name}\n")
    run_simulator(allocator_cls=allocator_cls)
