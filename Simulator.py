import numpy as np
import matplotlib.pyplot as plt

from SliceTask import SliceTask

class Simulator:
    def __init__(
        self,
        allocator,
        steps=100,
        arrival_rate=3  # mMTC avg rate; ~3 x avg size 2 = 6 PRB/step
    ):
        self.allocator = allocator
        self.steps = steps
        self.arrival_rate = arrival_rate

        self.time = 0
        self.requests = []

        self.slices = 3
        self.slice_names = ["eMBB", "URLLC", "mMTC"]
        self.slice_index = {s: i for i, s in enumerate(self.slice_names)}

        # per-slice task size range (inclusive) and deadline
        self.slice_config = {
            "eMBB":  {"size_range": (20, 50), "deadline": 80},
            "URLLC": {"size_range": (2, 8),   "deadline": 10},
            "mMTC":  {"size_range": (1, 3),   "deadline": 100},
        }

        # eMBB ON/OFF burst state — its own independent arrival process
        # duty cycle p_on = off_prob / (off_prob + (1-on_prob)) ≈ 0.17, so
        # avg rate ≈ 0.75 tasks/step × 35 avg size ≈ 26 PRB/step
        self.embb_on = True
        self.embb_on_prob = 0.5   # probability of staying ON
        self.embb_off_prob = 0.1  # probability of switching to ON from OFF
        self.embb_on_rate = 3     # arrivals/step while ON
        self.embb_off_rate = 0.3  # arrivals/step while OFF

        # URLLC periodic arrivals — deterministic, fixed batch every N steps
        # avg rate 0.25 tasks/step × 5 avg size ≈ 1.25 PRB/step
        self.urllc_period = 4      # steps between arrivals
        self.urllc_batch_size = 1  # tasks per arrival

        # highest arrival rate each slice can hit, used to size demand normalization below
        self.max_arrival_rate = {
            "eMBB": self.embb_on_rate,
            "URLLC": self.urllc_batch_size,
            "mMTC": arrival_rate,
        }

        # max expected per-slice demand = largest task size x highest arrival rate
        self.max_demand = np.array([
            self.slice_config[s]["size_range"][1] * self.max_arrival_rate[s]
            for s in self.slice_names
        ])

        self.demands_hist = []
        self.served_hist = []
        self.queue_hist = []
        self.alloc_hist = []
        self.reward_hist = []
        self.deadline_miss_hist = []
        self.avg_wait_hist = []
        self.state_hist = []
        self.starve_hist = []

        # consecutive steps a slice has had demand but received zero allocation
        self.starve_steps = np.zeros(3)

    def reset(self):
        self.time = 0
        self.requests = []
        self.embb_on = True
        self.demands_hist = []
        self.served_hist = []
        self.queue_hist = []
        self.alloc_hist = []
        self.reward_hist = []
        self.deadline_miss_hist = []
        self.avg_wait_hist = []
        self.state_hist = []
        self.starve_hist = []
        self.starve_steps = np.zeros(3)

    def generate_embb_arrivals(self):
        # transition state
        if self.embb_on:
            self.embb_on = np.random.rand() < self.embb_on_prob
        else:
            self.embb_on = np.random.rand() < self.embb_off_prob

        rate = self.embb_on_rate if self.embb_on else self.embb_off_rate
        return np.random.poisson(rate)

    def generate_urllc_arrivals(self):
        # deterministic periodic traffic: a fixed batch every N steps, none in between
        return self.urllc_batch_size if self.time % self.urllc_period == 0 else 0

    def generate_mmtc_arrivals(self):
        return np.random.poisson(self.arrival_rate)

    def make_task(self, slice_name):
        config = self.slice_config[slice_name]
        low, high = config["size_range"]
        size = np.random.randint(low, high + 1)  # +1 since np.random.randint's high is exclusive

        return SliceTask(slice_name, size, self.time, config["deadline"])

    def generate(self):
        arrivals = {
            "eMBB": self.generate_embb_arrivals(),
            "URLLC": self.generate_urllc_arrivals(),
            "mMTC": self.generate_mmtc_arrivals(),
        }

        for slice_name, n in arrivals.items():
            for _ in range(n):
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

    def compute_reward(self, served, demand, queue, alloc, deadline_miss, starve_steps):
        # throughput rate: fraction of demand actually served per slice (0–1)
        throughput_rate = np.divide(served, demand, out=np.zeros(3), where=demand > 0)

        # queue normalized against max expected backlog
        max_queue = self.arrival_rate * 20
        norm_queue = np.clip(queue / max_queue, 0, 1)

        # fraction of total PRBs actually used
        utilisation = alloc.sum() / self.allocator.total_prb

        # starvation: how long (relative to its own deadline) a slice has been
        # left with demand but under-served — squared so short neglect is cheap
        # but sustained abandonment is heavily punished. Clipped at 3x deadline
        # (not 1x): a 1x cap made the penalty flat/constant for most of a long
        # episode, giving the agent zero incentive to ever recover once starved.
        max_deadline = np.array([
            self.slice_config[s]["deadline"] for s in self.slice_names
        ])
        norm_starve = np.clip(starve_steps / max_deadline, 0, 3)

        # slice-specific weights (all inputs now 0–1 so weights are directly comparable)
        w_throughput = np.array([1.0, 0.5, 0.3])  # eMBB cares most about throughput
        w_deadline   = np.array([0.5, 1.5, 0.2])  # URLLC deadline miss penalized, but not so
                                                    # dominant it makes adapting to other slices pointless
        w_queue      = np.array([0.2, 0.8, 0.1])  # URLLC queue backlog is bad
        w_starve     = 1.0                        # fairness: same weight for every slice

        return (
            (w_throughput * throughput_rate).sum()
            + 0.5 * utilisation
            - (w_deadline * deadline_miss).sum()
            - (w_queue    * norm_queue).sum()
            - w_starve * (norm_starve ** 2).sum()
        )

    def get_state(self, demand, queue, deadline_miss, avg_wait):
        # Returns a 12-element normalized observation vector (all values in [0, 1]):
        #
        #   Indices  Signal                  Cap used
        #   -------  ----------------------  ---------------------------------
        #   0–2      demand per slice        max task size × max arrival rate (per slice)
        #   3–5      queue length per slice  arrival_rate × 20
        #   6–8      deadline miss rate      already 0–1
        #   9–11     avg wait per slice      per-slice deadlines
        #
        # Normalization caps derived from slice deadlines and traffic parameters
        max_queue    = self.arrival_rate * 20                       # generous backlog headroom
        max_deadline = np.array([
            self.slice_config[s]["deadline"] for s in self.slice_names
        ])

        norm_demand = np.clip(demand / self.max_demand, 0.0, 1.0)
        norm_queue  = np.clip(queue  / max_queue,  0.0, 1.0)
        norm_miss   = deadline_miss                                 # already 0–1
        norm_wait   = np.clip(avg_wait / max_deadline,  0.0, 1.0)

        return np.concatenate([norm_demand, norm_queue, norm_miss, norm_wait])

    def get_queue(self):
        q = np.zeros(3)
        for task in self.requests:
            if not task.is_complete():
                q[self.slice_index[task.slice_type]] += 1
        return q

    def get_deadline_miss_rate(self):
        misses = np.zeros(3)
        counts = np.zeros(3)
        for task in self.requests:
            if not task.is_complete():
                i = self.slice_index[task.slice_type]
                counts[i] += 1
                if task.is_deadline_missed(self.time):
                    misses[i] += 1
        return np.divide(misses, counts, out=np.zeros(3), where=counts > 0)

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
        current_state = np.zeros(12)  # initial observation before first step

        for t in range(self.steps):
            self.time = t

            self.generate()
            demand = self.aggregate_demand()

            # Action is chosen from the pre-serving state (proper MDP formulation)
            alloc = self.allocator.get_allocation(self.requests, current_state)
            served = self.serve(alloc.copy())

            queue         = self.get_queue()
            deadline_miss = self.get_deadline_miss_rate()
            avg_wait      = self.get_avg_wait()

            # update starvation streak: demand present but the slice received
            # under 10% of what it needed this step. NOTE: this must be based on
            # service ratio, not "alloc == 0" — the DQN action space enumerates
            # only splits with a,b,c >= 5, so every slice always gets >= 5 PRBs
            # and a literal zero-allocation check can never fire.
            service_ratio = np.divide(served, demand, out=np.ones(3), where=demand > 0)
            neglected = (demand > 0) & (service_ratio < 0.1)
            self.starve_steps = np.where(neglected, self.starve_steps + 1, 0)

            reward        = self.compute_reward(served, demand, queue, alloc, deadline_miss, self.starve_steps)
            next_state    = self.get_state(demand, queue, deadline_miss, avg_wait)

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
            self.deadline_miss_hist.append(deadline_miss)
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
            np.array(self.deadline_miss_hist),
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
        dm = np.array(self.deadline_miss_hist)
        al = np.array(self.alloc_hist)
        st = np.array(self.starve_hist)

        # throughput rate (served/demand) — same 0-1 scale for all three slices,
        # unlike raw served/arrived where eMBB's magnitude drowns out the others
        throughput_rate = np.divide(s, d, out=np.zeros_like(s, dtype=float), where=d > 0)

        labels = ["eMBB", "URLLC", "mMTC"]
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
            (axs[2], dm,              "SLA: Deadline Miss Rate",                        (-0.05, 1.05), False),
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