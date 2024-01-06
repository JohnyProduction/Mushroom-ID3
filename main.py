# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import plot_tree
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Load mushroom dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
data = pd.read_csv(url, header=None)
# Set column names for the dataset
data.columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

# Drop duplicates from the 'class' column (not modifying the DataFrame in-place)
data['class'].drop_duplicates()

# Encode 'class' column with LabelEncoder
le = LabelEncoder()
le.fit(data['class'].drop_duplicates().array)
le.classes_
data['class'] = le.transform(data['class'])

# Convert categorical columns to numerical codes
for col in data.columns:
    data[col] = pd.Categorical(data[col]).codes

# Split the data into training and testing sets
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Decision Tree model
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf = clf.fit(X_train, y_train)

tree.plot_tree(clf)

# Plot the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
plt.show()

# Make predictions and print the result
result = clf.predict(X_test.iloc[:1])
predicted_class = le.inverse_transform(result)
print(f"Predicted class for the first test sample: {predicted_class}")

# Evaluate the model with confusion matrix
y_result = clf.predict(X_test)
cm = confusion_matrix(y_result, y_test)
print("Confusion Matrix:")
print(cm)

# Confusion matrix plot
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp_tree.plot()
plt.show()
