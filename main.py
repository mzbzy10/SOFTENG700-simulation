import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Allocators"))

from FixedAllocator import FixedAllocator
from DQNAllocator import DQNAllocator
from Simulator import Simulator
from Environments import ENVIRONMENTS, SLICE_NAMES, offered_load

MODELS = [
    ("Fixed Allocator (baseline)", FixedAllocator),
    ("DQN Allocator (reinforcement learning)", DQNAllocator),
]

def select_from(prompt, options):
    # options: list of (label, value)
    print(f"\n{prompt}")
    for i, (label, _) in enumerate(options, start=1):
        print(f"  {i}. {label}")

    while True:
        choice = input(f"\nSelect (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")

def select_model():
    return select_from("Available models:", MODELS)

def select_environment():
    options = [
        (f"{key:<16} {env['description']}", key)
        for key, env in ENVIRONMENTS.items()
    ]
    return select_from("Available environments:", options)

def run_simulator(
    allocator_cls=FixedAllocator,
    env=None,
    episodes=150,
    steps=500
):
    if env is None:
        env = ENVIRONMENTS["balanced"]

    allocator = allocator_cls(total_prb=env["total_prb"])

    if isinstance(allocator, DQNAllocator):
        if input("\nLoad a pretrained model to continue training? (y/n): ").strip().lower() == 'y':
            path = input("Model path (default: dqn_model.pt): ").strip() or "dqn_model.pt"
            try:
                allocator.load(path)
            except FileNotFoundError:
                print(f"No model found at {path}, starting fresh.")

    sim = Simulator(
        allocator=allocator,
        env=env,
        steps=steps
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
    _, env_key = select_environment()
    name, allocator_cls = select_model()

    env = ENVIRONMENTS[env_key]
    load = offered_load(env)
    load_str = "  ".join(f"{s}={l:.1f}" for s, l in zip(SLICE_NAMES, load))

    print(f"\nEnvironment: {env_key}")
    print(f"Offered load (PRB/step): {load_str}  total={load.sum():.1f}")
    print(f"PRB budget: {env['total_prb']}")
    print(f"Running: {name}\n")

    run_simulator(allocator_cls=allocator_cls, env=env)
