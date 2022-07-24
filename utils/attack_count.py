import numpy as np


class AttackCountingFunction:
    def __init__(self, max_query):
        super(AttackCountingFunction, self).__init__()
        self.counts = []
        self.current_count = 0
        self.count_total = 0
        self.count_success = 0
        self.max_query = max_query

    def add(self, current_count, success):
        self.current_count = current_count
        # if not success:
        #     self.counts.append(self.max_query)
        # else:
        self.counts.append(self.current_count)
        self.count_total += 1
        self.count_success += int(success)

    def get_average(self, bound=1000, show_all=False):
        counts = np.array(self.counts)
        if show_all:
            print('seeL', counts)
        return np.mean(counts[counts < bound])

    def get_median(self, bound=1000):
        counts = np.array(self.counts)
        return np.median(counts[counts < bound])

    def get_first_success(self):
        counts = np.array(self.counts)
        return np.mean(counts == 1)

    def get_success_rate(self):
        return float(self.count_success) / self.count_total

    def get_all_counts(self):
        return self.counts
