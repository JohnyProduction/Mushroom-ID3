import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Wczytanie danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
data = pd.read_csv(url, header=None)
data.columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

# Zamiana symboli na liczby
for col in data.columns:
    data[col] = pd.Categorical(data[col]).codes

# Podział danych na zbiór uczący i testowy
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementacja algorytmu ID3 z parametrem max_depth
class Node:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None):
        self.feature = feature
        self.value = value
        self.results = results
        self.true_branch = true_branch
        self.false_branch = false_branch

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def build_tree(X, y, max_depth, current_depth=10):
    if len(set(y)) == 1:
        return Node(results=y.iloc[0])

    if X.empty or (max_depth is not None and current_depth == max_depth):
        return Node(results=y.mode()[0])

    current_entropy = entropy(y)
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    for column in X.columns:
        values = set(X[column])
        for value in values:
            true_data = X[X[column] == value]
            true_labels = y[X[column] == value]
            false_data = X[X[column] != value]
            false_labels = y[X[column] != value]

            if len(true_data) == 0 or len(false_data) == 0:
                continue

            p = len(true_data) / len(X)
            gain = current_entropy - p * entropy(true_labels) - (1 - p) * entropy(false_labels)

            if gain > best_gain:
                best_gain = gain
                best_criteria = (column, value)
                best_sets = (true_data, false_data, true_labels, false_labels)

    if best_gain > 0:
        true_branch = build_tree(best_sets[0], best_sets[2], max_depth, current_depth + 1)
        false_branch = build_tree(best_sets[1], best_sets[3], max_depth, current_depth + 1)
        return Node(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)
    else:
        return Node(results=y.mode()[0])

def print_tree(node, spacing=""):
    if node.results is not None:
        print(spacing + "Predict:", node.results)
        return

    print(spacing + f"{node.feature} = {node.value}?")
    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + "  ")

# Budowanie drzewa decyzyjnego z ograniczeniem głębokości do max_depth
max_depth = 12
tree = build_tree(X_train, y_train, max_depth)

# Wyświetlanie drzewa decyzyjnego
print_tree(tree)

# Klasyfikacja na zbiorze testowym
def classify(row, node):
    if node.results is not None:
        return node.results

    value = row[node.feature]
    branch = None

    if isinstance(value, (int, float)):
        branch = node.true_branch if value >= node.value else node.false_branch
    else:
        branch = node.true_branch if value == node.value else node.false_branch

    return classify(row, branch)

# Ocena dokładności klasyfikatora na zbiorze testowym
correct = 0
for i in range(len(X_test)):
    if classify(X_test.iloc[i], tree) == y_test.iloc[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"Accuracy: {accuracy:.4f}")
