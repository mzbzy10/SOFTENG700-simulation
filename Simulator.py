from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from SliceTask import SliceTask
from Environments import (
    SLICE_NAMES,
    ARRIVAL_MODELS,
    SLA,
    NORM_MAX_DEMAND,
    NORM_MAX_QUEUE,
    NORM_MAX_WAIT,
)
import Reward

class Simulator:
    def __init__(
        self,
        allocator,
        env,
        steps=100
    ):
        self.allocator = allocator
        self.env = env
        self.steps = steps

        self.time = 0
        self.requests = []

        self.slice_names = list(SLICE_NAMES)
        self.slices = len(self.slice_names)
        self.slice_index = {s: i for i, s in enumerate(self.slice_names)}

        # per-slice task size range (inclusive), from the environment
        self.slice_config = env["slices"]
        self.arrival_config = env["arrivals"]

        # ON/OFF arrival processes carry state between steps; every slice gets an
        # entry so any slice can use the "onoff" model
        self.onoff_state = {s: True for s in self.slice_names}

        # eMBB's KPI is a sustained data rate, so served/demand are averaged over
        # a trailing window rather than judged per step
        self.rate_window = SLA["eMBB"]["window"]
        self.served_window = deque(maxlen=self.rate_window)
        self.demand_window = deque(maxlen=self.rate_window)

        self.demands_hist = []
        self.served_hist = []
        self.queue_hist = []
        self.alloc_hist = []
        self.reward_hist = []
        self.ssr_hist = []
        self.avg_wait_hist = []
        self.state_hist = []
        self.starve_hist = []

        # consecutive steps a slice has had demand but received zero allocation
        self.starve_steps = np.zeros(3)

    def reset(self):
        self.time = 0
        self.requests = []
        self.onoff_state = {s: True for s in self.slice_names}
        self.served_window = deque(maxlen=self.rate_window)
        self.demand_window = deque(maxlen=self.rate_window)
        self.demands_hist = []
        self.served_hist = []
        self.queue_hist = []
        self.alloc_hist = []
        self.reward_hist = []
        self.ssr_hist = []
        self.avg_wait_hist = []
        self.state_hist = []
        self.starve_hist = []
        self.starve_steps = np.zeros(3)

    def generate_arrivals(self, slice_name):
        # Number of tasks arriving for one slice this step. The process is fixed
        # per slice (Environments.ARRIVAL_MODELS); environments only tune its
        # parameters.
        cfg = self.arrival_config[slice_name]
        model = ARRIVAL_MODELS[slice_name]

        if model == "poisson":
            return np.random.poisson(cfg["rate"])

        if model == "onoff":
            # two-state Markov-modulated Poisson process, for bursty traffic
            if self.onoff_state[slice_name]:
                on = np.random.rand() < cfg["on_prob"]
            else:
                on = np.random.rand() < cfg["off_prob"]
            self.onoff_state[slice_name] = on

            return np.random.poisson(cfg["on_rate"] if on else cfg["off_rate"])

        if model == "periodic":
            # deterministic: a fixed batch every N steps, none in between
            return cfg["batch_size"] if self.time % cfg["period"] == 0 else 0

        raise ValueError(f"unknown arrival model for {slice_name}: {model}")

    def make_task(self, slice_name):
        low, high = self.slice_config[slice_name]["size_range"]
        size = np.random.randint(low, high + 1)  # +1 since np.random.randint's high is exclusive

        return SliceTask(slice_name, size, self.time)

    def generate(self):
        for slice_name in self.slice_names:
            for _ in range(self.generate_arrivals(slice_name)):
                self.requests.append(self.make_task(slice_name))

    def aggregate_demand(self):
        d = np.zeros(3)

        for task in self.requests:
            d[self.slice_index[task.slice_type]] += task.remaining

        return d

    def serve(self, alloc):
        served = np.zeros(3)
        remaining_cap = alloc.copy().astype(float)

        for task in self.requests:
            i = self.slice_index[task.slice_type]
            if remaining_cap[i] > 0:
                used = task.serve(remaining_cap[i])
                served[i] += used
                remaining_cap[i] -= used

        self.requests = [
            task for task in self.requests if not task.is_complete()
        ]

        return served

    def compute_reward(self, served, demand, queue, alloc, ssr, starve_steps):
        # the reward algorithm and all its weights live in Reward.py
        return Reward.compute_reward(
            served, demand, queue, alloc, ssr, starve_steps,
            total_prb=self.allocator.total_prb,
        )

    def get_state(self, demand, queue, ssr, avg_wait):
        # Returns a 12-element normalized observation vector (all values in [0, 1]):
        #
        #   Indices  Signal                  Cap used
        #   -------  ----------------------  ------------------
        #   0–2      demand per slice        NORM_MAX_DEMAND
        #   3–5      queue length per slice  NORM_MAX_QUEUE
        #   6–8      SLA satisfaction ratio  already 0–1
        #   9–11     avg wait per slice      NORM_MAX_WAIT
        #
        # The caps are global constants, not per-environment — see the warning in
        # Environments/__init__.py. Scaling the observation differently in each
        # environment would corrupt any cross-environment transfer measurement.
        norm_demand = np.clip(demand   / NORM_MAX_DEMAND, 0.0, 1.0)
        norm_queue  = np.clip(queue    / NORM_MAX_QUEUE,  0.0, 1.0)
        norm_ssr    = ssr                                   # already 0–1
        norm_wait   = np.clip(avg_wait / NORM_MAX_WAIT,   0.0, 1.0)

        return np.concatenate([norm_demand, norm_queue, norm_ssr, norm_wait])

    def get_queue(self):
        q = np.zeros(3)
        for task in self.requests:
            if not task.is_complete():
                q[self.slice_index[task.slice_type]] += 1
        return q

    def get_ssr(self, served, demand, queue):
        """Per-slice SLA satisfaction ratio for this step, each on its own KPI.

        One KPI per slice type rather than one metric with three thresholds —
        see the SLA block in Environments/__init__.py for the rationale and
        citations. Every component is in [0, 1] and 1.0 means fully satisfied,
        so the three remain directly comparable and summable in the reward.
        """
        ssr = np.ones(3)

        # eMBB — minimum data rate, judged over a trailing window because a data
        # rate is inherently a rate over time. Compared against what was
        # *achievable*: during a lull the slice cannot be served at min_rate, and
        # is not held to it.
        i = self.slice_index["eMBB"]
        served_w = sum(s[i] for s in self.served_window)
        demand_w = sum(d[i] for d in self.demand_window)
        target = min(SLA["eMBB"]["min_rate"] * len(self.served_window), demand_w)
        ssr[i] = 1.0 if target <= 0 else np.clip(served_w / target, 0.0, 1.0)

        # URLLC — maximum delay: fraction of in-system tasks still within budget
        i = self.slice_index["URLLC"]
        budget = SLA["URLLC"]["max_delay"]
        held = [t for t in self.requests if t.slice_type == "URLLC"]
        if held:
            ontime = sum(1 for t in held if t.waiting_time(self.time) <= budget)
            ssr[i] = ontime / len(held)

        # mMTC — maximum buffer: satisfied while backlog is within the cap, then
        # decaying as the ratio by which it is exceeded (2x over -> 0.5)
        i = self.slice_index["mMTC"]
        cap = SLA["mMTC"]["max_buffer"]
        ssr[i] = 1.0 if queue[i] <= cap else np.clip(cap / queue[i], 0.0, 1.0)

        return ssr

    def get_avg_wait(self):
        totals = np.zeros(3)
        counts = np.zeros(3)
        for task in self.requests:
            if not task.is_complete():
                i = self.slice_index[task.slice_type]
                counts[i] += 1
                totals[i] += task.waiting_time(self.time)
        return np.divide(totals, counts, out=np.zeros(3), where=counts > 0)

    def run(self, target_update_freq=10):
        # initial observation before the first step: 4 signals x 3 slices
        current_state = np.zeros(4 * self.slices)

        for t in range(self.steps):
            self.time = t

            self.generate()
            demand = self.aggregate_demand()

            # Action is chosen from the pre-serving state (proper MDP formulation)
            alloc = self.allocator.get_allocation(self.requests, current_state)
            served = self.serve(alloc.copy())

            queue    = self.get_queue()
            avg_wait = self.get_avg_wait()

            # eMBB's rate KPI needs a trailing window; push this step before
            # evaluating SSR so the current step counts toward it
            self.served_window.append(served)
            self.demand_window.append(demand)
            ssr = self.get_ssr(served, demand, queue)

            # update starvation streak: demand present but the slice was served
            # less than Reward.STARVE_SERVICE_THRESHOLD of what it needed
            service_ratio = np.divide(served, demand, out=np.ones(3), where=demand > 0)
            neglected = (demand > 0) & (service_ratio < Reward.STARVE_SERVICE_THRESHOLD)
            self.starve_steps = np.where(neglected, self.starve_steps + 1, 0)

            reward     = self.compute_reward(served, demand, queue, alloc, ssr, self.starve_steps)
            next_state = self.get_state(demand, queue, ssr, avg_wait)

            # RL training hooks — no-ops for non-RL allocators
            done = (t == self.steps - 1)
            if hasattr(self.allocator, 'store') and self.allocator.last_action_idx is not None:
                self.allocator.store(current_state, self.allocator.last_action_idx, reward, next_state, done)
            if hasattr(self.allocator, 'train_step'):
                self.allocator.train_step()
            if hasattr(self.allocator, 'update_target') and t % target_update_freq == 0:
                self.allocator.update_target()

            self.demands_hist.append(demand)
            self.served_hist.append(served)
            self.queue_hist.append(queue)
            self.alloc_hist.append(alloc)
            self.reward_hist.append(reward)
            self.ssr_hist.append(ssr)
            self.avg_wait_hist.append(avg_wait)
            self.state_hist.append(next_state)
            self.starve_hist.append(self.starve_steps.copy())

            current_state = next_state

        return (
            np.array(self.demands_hist),
            np.array(self.served_hist),
            np.array(self.queue_hist),
            np.array(self.alloc_hist),
            np.array(self.reward_hist),
            np.array(self.ssr_hist),
            np.array(self.avg_wait_hist)
        )

    @staticmethod
    def _smooth(data, window=20, max_points=150):
        # rolling mean per column + downsampling, so fast-oscillating per-step
        # signals (the DQN can switch actions every step) read as a trend
        # instead of a solid block of color
        n = len(data)
        w = min(window, n)
        if w <= 1:
            return data, np.arange(n)

        kernel = np.ones(w) / w
        smoothed = np.stack(
            [np.convolve(data[:, i], kernel, mode='valid') for i in range(data.shape[1])],
            axis=1
        )
        xs = np.arange(w - 1, n)

        stride = max(1, len(xs) // max_points)
        return smoothed[::stride], xs[::stride]

    def visualize(self, episode_rewards=None):
        d  = np.array(self.demands_hist)
        s  = np.array(self.served_hist)
        r  = np.array(self.reward_hist)
        sr = np.array(self.ssr_hist)
        al = np.array(self.alloc_hist)
        st = np.array(self.starve_hist)

        # throughput rate (served/demand) — same 0-1 scale for all three slices,
        # unlike raw served/arrived where eMBB's magnitude drowns out the others
        throughput_rate = np.divide(s, d, out=np.zeros_like(s, dtype=float), where=d > 0)

        labels = self.slice_names
        colors = ["tab:blue", "tab:green", "tab:orange"]
        has_episodes = episode_rewards is not None and len(episode_rewards) > 1

        fig, axs = plt.subplots(3, 2, figsize=(14, 11))
        axs = axs.flatten()

        # top-left: training progress (reward objective as a whole)
        ax = axs[0]
        if has_episodes:
            ax.plot(range(1, len(episode_rewards) + 1), episode_rewards,
                     marker='o', markersize=3, alpha=0.35, label="episode reward")
            window = min(5, len(episode_rewards))
            if window > 1:
                rolling = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
                ax.plot(range(window, len(episode_rewards) + 1), rolling,
                         color='red', label=f"{window}-episode avg")
            ax.set_title("Total Reward per Episode")
            ax.set_xlabel("Episode")
            ax.legend()
        else:
            ax.plot(r)
            ax.set_title("Reward per Step")
            ax.set_xlabel("Step")

        # remaining panels: reward broken into the four things it actually optimizes for
        # (throughput and allocation are smoothed — raw per-step values thrash
        # too fast to read; deadline-miss and starvation are already slow-moving)
        panels = [
            (axs[1], throughput_rate, "Throughput Rate (served / demand, 20-step avg)", (-0.05, 1.05), True),
            (axs[2], sr,              "SLA Satisfaction Ratio (per-slice KPI)",         (-0.05, 1.05), True),
            (axs[3], al,              "Resource: PRB Allocation (20-step avg)",         None,           True),
            (axs[4], st,              "Fairness: Starvation Streak (steps neglected)",  None,           False),
        ]
        for ax, data, title, ylim, smooth in panels:
            if smooth:
                plot_data, xs = self._smooth(data)
            else:
                plot_data, xs = data, np.arange(len(data))

            for i in range(self.slices):
                ax.plot(xs, plot_data[:, i], color=colors[i], label=labels[i])
            ax.set_title(title)
            ax.set_xlabel("Step")
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend()

        axs[5].axis('off')

        plt.tight_layout()
        plt.show()