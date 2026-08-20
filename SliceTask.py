class SliceTask:
    """One arriving task for one slice.

    Tasks no longer carry a deadline: SLA is defined per slice *type* rather
    than per task (see Environments.SLA), and only URLLC's KPI is delay-based.
    """

    def __init__(self, slice_type, size, arrival_time):
        self.slice_type = slice_type
        self.size = size
        self.remaining = size
        self.arrival_time = arrival_time

    def serve(self, amount):
        used = min(amount, self.remaining)
        self.remaining -= used
        return used

    def is_complete(self):
        return self.remaining <= 0

    def waiting_time(self, current_time):
        return current_time - self.arrival_time
