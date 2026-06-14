import numpy as np

from SliceTask import SliceTask


class Request:
    _id = 0

    def __init__(self, arrival_time, job_size_mean):
        self.id = Request._id
        Request._id += 1

        self.time = arrival_time

        ## define tasks for each slice with different sizes and deadlines
        ## first number is the size, second number is the deadline (time by which it should be completed)
        self.tasks = {
            "eMBB": SliceTask("eMBB", np.random.randint(job_size_mean, job_size_mean * 3), arrival_time, 80),
            "URLLC": SliceTask("URLLC", np.random.randint(1, job_size_mean), arrival_time, 10),
            "mMTC": SliceTask("mMTC", np.random.randint(1, job_size_mean // 2 + 1), arrival_time, 100),
        }

    def is_complete(self):
        return all(t.is_complete() for t in self.tasks.values())