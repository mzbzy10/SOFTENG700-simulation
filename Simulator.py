import numpy as np
import matplotlib.pyplot as plt

from Request import Request

class Simulator:
    def __init__(
        self,
        allocator,
        steps=100,
        arrival_rate=5,
        job_size_mean=10
    ):
        self.allocator = allocator
        self.steps = steps
        self.arrival_rate = arrival_rate
        self.job_size_mean = job_size_mean

        self.time = 0
        self.requests = []

        self.slices = 3
        self.slice_names = ["eMBB", "URLLC", "mMTC"]

        self.demands_hist = []
        self.served_hist = []
        self.queue_hist = []
        self.alloc_hist = []
        self.reward_hist = []
        self.deadline_miss_hist = []
        self.avg_wait_hist = []

    def generate(self):
        n = np.random.poisson(self.arrival_rate)

        for _ in range(n):
            self.requests.append(
                Request(self.time, self.job_size_mean)
            )

    def aggregate_demand(self):
        d = np.zeros(3)

        for r in self.requests:
            for i, s in enumerate(self.slice_names):
                d[i] += r.tasks[s].remaining

        return d

    def serve(self, alloc):
        served = np.zeros(3)
        remaining_cap = alloc.copy().astype(float)

        for r in self.requests:
            for i, s in enumerate(self.slice_names):
                if remaining_cap[i] > 0:
                    used = r.tasks[s].serve(remaining_cap[i])
                    served[i] += used
                    remaining_cap[i] -= used

        self.requests = [
            r for r in self.requests if not r.is_complete()
        ]

        return served

    def compute_reward(self, served, queue, deadline_miss, avg_wait):
        # Per-slice weights reflecting 5G slice SLA priorities
        w_served   = np.array([1.0, 0.8, 0.5])   # eMBB: throughput; URLLC: moderate; mMTC: low
        w_queue    = np.array([0.2, 0.5, 0.1])   # URLLC backlog is most damaging
        w_deadline = np.array([0.5, 2.0, 0.3])   # URLLC deadline=5 → heavy miss penalty
        w_wait     = np.array([0.05, 0.3, 0.02]) # URLLC latency SLA is strictest

        return (
            (w_served   * served).sum()
            - (w_queue    * queue).sum()
            - (w_deadline * deadline_miss).sum()
            - (w_wait     * avg_wait).sum()
        )

    def get_queue(self):
        q = np.zeros(3)
        for r in self.requests:
            for i, s in enumerate(self.slice_names):
                if not r.tasks[s].is_complete():
                    q[i] += 1
        return q

    def get_deadline_miss_rate(self):
        misses = np.zeros(3)
        counts = np.zeros(3)
        for r in self.requests:
            for i, s in enumerate(self.slice_names):
                if not r.tasks[s].is_complete():
                    counts[i] += 1
                    if r.tasks[s].is_deadline_missed(self.time):
                        misses[i] += 1
        return np.divide(misses, counts, out=np.zeros(3), where=counts > 0)

    def get_avg_wait(self):
        totals = np.zeros(3)
        counts = np.zeros(3)
        for r in self.requests:
            for i, s in enumerate(self.slice_names):
                if not r.tasks[s].is_complete():
                    counts[i] += 1
                    totals[i] += r.tasks[s].waiting_time(self.time)
        return np.divide(totals, counts, out=np.zeros(3), where=counts > 0)

    def run(self):
        for t in range(self.steps):
            self.time = t

            self.generate()

            demand = self.aggregate_demand()

            alloc = self.allocator.get_allocation(self.requests)

            served = self.serve(alloc.copy())

            queue = self.get_queue()
            deadline_miss = self.get_deadline_miss_rate()
            avg_wait = self.get_avg_wait()

            reward = self.compute_reward(served, queue, deadline_miss, avg_wait)

            self.demands_hist.append(demand)
            self.served_hist.append(served)
            self.queue_hist.append(queue)
            self.alloc_hist.append(alloc)
            self.reward_hist.append(reward)
            self.deadline_miss_hist.append(deadline_miss)
            self.avg_wait_hist.append(avg_wait)

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