import numpy as np
from typing import List, Optional, Tuple
from collections import Counter


class DecisionNode:
    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["DecisionNode"] = None,
        right: Optional["DecisionNode"] = None,
        value: Optional[int] = None,
    ):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class label if leaf node


class DecisionTreeClassifier:
    def __init__(self, max_deep, min_sample_leaft):
        self.max_deep = max_deep
        self.min_sample_leaft = min_sample_leaft

    def build_tree(self):
        pass

    def _split_data(self, x: np.ndarray, threshold: float | int):
        left_idxs = np.where(x <= threshold)[0]
        right_idxs = np.where(x > threshold)[0]
        return left_idxs, right_idxs

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_serie = X[:, feat_idx]
            thresholds = np.unique(X_serie)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(X_serie, y, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _entropy(self, y):
        probabilitis = np.bincount(y) / len(y)
        return sum([p * np.log2(p) for p in probabilitis if p > 0])

    def _information_gain(self, x: np.ndarray, y: np.ndarray, threshold: float | int):
        information_gain = self._entropy(y)

        n_total = len(y)

        for idxs in self._split_data(x, threshold):
            n_subset = len(idxs)

            if len(n_subset) == 0:
                return 0

            child_entropy = (n_subset / n_total) * self._entropy(y[idxs])

            information_gain -= child_entropy

        return information_gain
