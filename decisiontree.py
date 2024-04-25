from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        *,
        value=None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        features=None
    ) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.features = features
        self.root = None

    def _entropy(self, y):
        count = np.bincount(y)
        p = count / len(y)
        return -np.sum([i * np.log(i) for i in p if i > 0])

    def _split(self, X_column, split_threshold):
        left = np.argwhere(X_column <= split_threshold).flatten()
        right = np.argwhere(X_column > split_threshold).flatten()
        return left, right

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left, right = self._split(X_column, threshold)
        if len(left) == 0 or len(right) == 0:
            return 0

        length = len(y)
        left_length, right_length = len(left), len(right)
        left_entropy, right_entropy = self._entropy(y[left]), self._entropy(y[right])
        child_entropy = (left_length / length) * left_entropy \
            + (right_length / length) * right_entropy

        return parent_entropy - child_entropy

    def _best_split(self, X, y, rfeatures):
        best_gain = -1
        split_feature, split_threshold = None, None

        for feature in rfeatures:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_threshold = threshold

        return split_feature, split_threshold

    def _grow_tree(self, X, y, depth=0) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            counter = Counter(y)
            leaf_value = counter.most_common(1)[0][0]
            return Node(value=leaf_value)

        rfeatures = np.random.choice(n_features, self.features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, rfeatures)

        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idx,:], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx,:], y[right_idx], depth+1)
        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y) -> None:
        self.features = X.shape[1] if not self.features else min(X.shape[1], self.features)
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x, node: Node):
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

classifier = DecisionTree()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
