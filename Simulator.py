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

    def compute_reward(self, served, queue):
        return served.sum() - 0.1 * queue.sum()

    def get_queue(self):
      q = np.zeros(3)
      for r in self.requests:
          for i, s in enumerate(self.slice_names):
              if not r.tasks[s].is_complete():
                  q[i] += 1
      return q



    def run(self):
        for t in range(self.steps):
            self.time = t

            self.generate()

            demand = self.aggregate_demand()

            alloc = self.allocator.get_allocation(self.requests)

            served = self.serve(alloc.copy())

            queue = self.get_queue()

            reward = self.compute_reward(served, queue)

            self.demands_hist.append(demand)
            self.served_hist.append(served)
            self.queue_hist.append(queue)
            self.alloc_hist.append(alloc)
            self.reward_hist.append(reward)

        return (
            np.array(self.demands_hist),
            np.array(self.served_hist),
            np.array(self.queue_hist),
            np.array(self.alloc_hist),
            np.array(self.reward_hist)
        )

    def visualize(self):
        d = np.array(self.demands_hist)
        s = np.array(self.served_hist)
        q = np.array(self.queue_hist)
        a = np.array(self.alloc_hist)
        r = np.array(self.reward_hist)

        labels = ["eMBB", "URLLC", "mMTC"]

        fig, axs = plt.subplots(5, 1, figsize=(12, 12))

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

        plt.tight_layout()
        plt.show()