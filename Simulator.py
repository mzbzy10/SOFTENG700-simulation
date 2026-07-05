import numpy as np
import matplotlib.pyplot as plt

from SliceTask import SliceTask

class Simulator:
    def __init__(
        self,
        allocator,
        steps=100,
        arrival_rate=5
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
        self.embb_on = True
        self.embb_on_prob = 0.7   # probability of staying ON
        self.embb_off_prob = 0.3  # probability of switching to ON from OFF
        self.embb_on_rate = 8     # arrivals/step while ON
        self.embb_off_rate = 1    # arrivals/step while OFF

        # URLLC periodic arrivals — deterministic, fixed batch every N steps
        self.urllc_period = 2      # steps between arrivals
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

    def compute_reward(self, served, demand, queue, alloc, deadline_miss):
        # throughput rate: fraction of demand actually served per slice (0–1)
        throughput_rate = np.divide(served, demand, out=np.zeros(3), where=demand > 0)

        # queue normalized against max expected backlog
        max_queue = self.arrival_rate * 20
        norm_queue = np.clip(queue / max_queue, 0, 1)

        # fraction of total PRBs actually used
        utilisation = alloc.sum() / self.allocator.total_prb

        # slice-specific weights (all inputs now 0–1 so weights are directly comparable)
        w_throughput = np.array([1.0, 0.5, 0.3])  # eMBB cares most about throughput
        w_deadline   = np.array([0.5, 3.0, 0.2])  # URLLC deadline miss is heavily penalized
        w_queue      = np.array([0.2, 0.8, 0.1])  # URLLC queue backlog is bad

        return (
            (w_throughput * throughput_rate).sum()
            + 0.5 * utilisation
            - (w_deadline * deadline_miss).sum()
            - (w_queue    * norm_queue).sum()
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
            reward        = self.compute_reward(served, demand, queue, alloc, deadline_miss)
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

    def visualize(self):
        d = np.array(self.demands_hist)
        s = np.array(self.served_hist)
        q = np.array(self.queue_hist)
        a = np.array(self.alloc_hist)
        r = np.array(self.reward_hist)
        dm = np.array(self.deadline_miss_hist)
        aw = np.array(self.avg_wait_hist)

        labels = ["eMBB", "URLLC", "mMTC"]

        fig, axs = plt.subplots(7, 1, figsize=(12, 16))

        for i in range(self.slices):
            axs[0].plot(d[:, i], label=labels[i])
        axs[0].set_title("Demand")
        axs[0].legend()

        for i in range(self.slices):
            axs[1].plot(s[:, i], label=labels[i])
        axs[1].set_title("Served")
        axs[1].legend()

        for i in range(self.slices):
            axs[2].plot(q[:, i], label=labels[i])
        axs[2].set_title("Queue Length")
        axs[2].legend()

        for i in range(self.slices):
            axs[3].plot(a[:, i], label=labels[i])
        axs[3].set_title("PRB Allocation")
        axs[3].legend()

        t = np.arange(len(r))
        axs[4].plot(t, r)
        axs[4].set_title("Reward")

        for i in range(self.slices):
            axs[5].plot(dm[:, i], label=labels[i])
        axs[5].set_title("Deadline Miss Rate")
        axs[5].set_ylabel("Fraction")
        axs[5].legend()

        for i in range(self.slices):
            axs[6].plot(aw[:, i], label=labels[i])
        axs[6].set_title("Average Waiting Time per Slice")
        axs[6].set_ylabel("Time steps")
        axs[6].legend()

        plt.tight_layout()
        plt.show()