import numpy as np
import matplotlib.pyplot as plt

class SliceTask:
    def __init__(self, slice_type, size, arrival_time, deadline):
        self.slice_type = slice_type
        self.size = size
        self.remaining = size
        self.arrival_time = arrival_time
        self.deadline = deadline

    def serve(self, amount):
        used = min(amount, self.remaining)
        self.remaining -= used
        return used

    def is_complete(self):
        return self.remaining <= 0

    def waiting_time(self, current_time):
        return current_time - self.arrival_time

    def is_deadline_missed(self, current_time):
        return self.waiting_time(current_time) > self.deadline

    def deadline_missed_by(self, current_time):
        return self.waiting_time(current_time) - self.deadline