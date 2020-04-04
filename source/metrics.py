import numpy as np


class Accuracy(object):
    def __init__(self):
        self.total = 0
        self.right = 0

    def add_sample(self, pred_y, y):
        self.total += 1
        if pred_y == y:
            self.right += 1

    def reset(self):
        self.total = 0
        self.right = 0

    def calculate(self, is_clear=True):
        acc = self.right / self.total
        if is_clear:
            self.reset()
        return acc


class MacroF1(object):
    def __init__(self, num_label):
        self.num_label = num_label
        self.f1_count = {"tp": np.zeros([num_label]),
                         "fn": np.zeros([num_label]),
                         "fp": np.zeros([num_label])}

    def add_sample(self, pred_y, y):
        if pred_y == y:
            self.f1_count["tp"][y] += 1
        else:
            self.f1_count["fp"][pred_y] += 1
            self.f1_count["fn"][y] += 1

    def reset(self, num_label=None):
        del self.f1_count
        if num_label is None:
            num_label = self.num_label
        self.f1_count = {"tp": np.zeros([num_label]),
                         "fn": np.zeros([num_label]),
                         "fp": np.zeros([num_label])}

    def calculate(self, is_clear=True):
        precision = self.f1_count["tp"] / (self.f1_count["tp"] + self.f1_count["fp"])
        recall = self.f1_count["tp"] / (self.f1_count["tp"] + self.f1_count["fn"])
        f1 = 2 * np.mean(np.multiply(precision, recall) / (precision + recall))

        if is_clear:
            self.reset()

        return f1
