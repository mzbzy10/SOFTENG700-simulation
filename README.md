# SOFTENG700 Network Slicing Simulation

A discrete-time simulation of Physical Resource Block (PRB) allocation across
three 5G network slices — **eMBB**, **URLLC**, and **mMTC** — used to compare
a fixed allocation policy against a Deep Q-Network (DQN) reinforcement
learning policy.

## Overview

Each simulation step:

1. New tasks arrive independently for each slice (see [Arrival model](#arrival-model)).
2. The allocator observes the current state and splits the total PRB budget
   across the three slices.
3. Queued tasks are served up to their slice's allocated capacity (FIFO per slice).
4. Reward, queue length, deadline miss rate, and average wait time are computed
   and fed back to the allocator (for RL allocators) as the next observation.

## Project structure

```
main.py                     Entry point: pick an allocator, run episodes, plot results
Simulator.py                Core simulation loop, arrivals, reward, and state
SliceTask.py                A single arriving task for one slice (size, deadline, progress)
Allocators/
  FixedAllocator.py         Static proportional PRB split (baseline)
  DQNAllocator.py           DQN-based allocator with experience replay + target network
requirements.txt            Python dependencies
dqn_model.pt                Saved DQN checkpoint (optional, created via main.py)
```

## Slices

| Slice | Task size range              | Deadline | Notes                          |
|-------|-------------------------------|----------|---------------------------------|
| eMBB  | `[job_size_mean, 3×job_size_mean)` | 80 steps | Large jobs, throughput-focused |
| URLLC | `[1, job_size_mean)`           | 10 steps | Small jobs, latency-critical    |
| mMTC  | `[1, job_size_mean/2 + 1)`     | 100 steps| Small jobs, loose deadline      |

## Arrival model

Each slice generates its own arrivals independently every step — tasks are no
longer bundled together across slices.

- **eMBB** uses a two-state ON/OFF Markov-modulated Poisson process to model
  bursty traffic:
  - ON → stays ON with probability `embb_on_prob` (0.7), arrival rate `8`
  - OFF → switches ON with probability `embb_off_prob` (0.3), arrival rate `1`
- **URLLC** and **mMTC** currently draw from `Poisson(arrival_rate)` each step
  (placeholder — swap the bodies of `generate_urllc_arrivals` /
  `generate_mmtc_arrivals` in `Simulator.py` for other traffic models).

## State and reward

`Simulator.get_state()` returns a 12-element normalized vector:

| Indices | Signal                  |
|---------|--------------------------|
| 0–2     | Demand per slice         |
| 3–5     | Queue length per slice   |
| 6–8     | Deadline miss rate       |
| 9–11    | Average wait time per slice |

`Simulator.compute_reward()` combines per-slice throughput, PRB utilisation,
deadline miss penalties, and queue backlog penalties, with weights tuned so
URLLC deadline misses and queue backlog are penalized most heavily.

## Allocators

- **FixedAllocator** — splits the PRB budget by a fixed ratio (`0.5 / 0.3 / 0.2`
  for eMBB / URLLC / mMTC). Used as a baseline.
- **DQNAllocator** — chooses from a discretized set of PRB splits (multiples
  of 5) using a Q-network trained online via experience replay and a target
  network. Supports `save()` / `load()` for checkpointing.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3 with `numpy`, `matplotlib`, and `torch`.

## Running

```bash
python main.py
```

You'll be prompted to choose an allocator (Fixed or DQN), then the simulation
runs for a configurable number of episodes and steps (see `run_simulator()` in
`main.py`), plotting:

- Total reward per episode
- Per-slice demand, served load, queue length, PRB allocation
- Reward, deadline miss rate, and average wait time over time

If you run the DQN allocator, you'll be asked at the end whether to save the
trained model (default path: `dqn_model.pt`).
