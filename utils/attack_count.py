import numpy as np


class AttackCountingFunction:
    def __init__(self, max_query):
        super(AttackCountingFunction, self).__init__()
        self.all_counts = []
        self.success_list = []
        self.success_counts = []

        self.current_count = 0

        self.count_total = 0
        self.count_success = 0

        self.max_query = max_query

    def add(self, current_count, success):
        self.current_count = current_count
        self.all_counts.append(self.current_count)
        self.success_list.append(success)
        if success:
            self.success_counts.append(current_count)

    def get_average(self, show_all=False):
        if show_all:
            print('Counts', self.all_counts)
            print('Success', self.success_list)
        counts = np.array(self.success_counts)
        if len(counts) == 0:
            counts = np.array([0])
        return np.mean(counts)

    def get_median(self):
        counts = np.array(self.success_list)
        if len(counts) == 0:
            counts = np.array([0])
        return np.median(counts)

    def get_first_success(self):
        counts = np.array(self.success_counts)
        if len(counts) == 0:
            counts = np.array([0])
        return np.mean(counts == 1)

    def get_success_rate(self):
        return float(len(self.success_counts)) / len(self.all_counts)

    def get_all_counts(self):
        return self.all_counts
