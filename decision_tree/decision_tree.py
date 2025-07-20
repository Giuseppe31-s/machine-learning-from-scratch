import numpy as np
from typing import List, Optional, Tuple
from scipy.stats import mode
from collections import Counter


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 0, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = 0

    def fit(self, X, y):
        self.n_features = (
            X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        )
        self.root = self.build_tree(X, y)

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split_data(X[:, best_feature], best_thresh)
        left = self.build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _split_data(self, x: np.ndarray, threshold: float | int):
        left_idxs = np.where(x <= threshold)[0]
        right_idxs = np.where(x > threshold)[0]
        return left_idxs, right_idxs

    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idxs: list[int]):
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

    def _entropy(self, y: np.ndarray):
        probabilitis = np.bincount(y) / len(y)
        return sum([p * np.log2(p) for p in probabilitis if p > 0])

    def _information_gain(self, x: np.ndarray, y: np.ndarray, threshold: float | int):
        information_gain = self._entropy(y)

        n_total = len(y)

        for idxs in self._split_data(x, threshold):
            n_subset = len(idxs)

            if n_subset == 0:
                return 0

            child_entropy = (n_subset / n_total) * self._entropy(y[idxs])

            information_gain -= child_entropy

        return information_gain

    def _most_common_label(self, y: np.ndarray):
        if y.shape[0] > 10_000:
            value = mode(y, nan_policy="omit").mode
        else:
            value = Counter(y).most_common(1)[0][0]
        return value

    def predict(self, X: np.ndarray):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
