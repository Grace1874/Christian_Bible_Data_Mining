
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

project_folder = Path(__file__).resolve().parent.parent

# Load training and testing datasets
train = pd.read_csv(project_folder / "Results" / "tree_train.csv")
test = pd.read_csv(project_folder / "Results" / "tree_test.csv")

X_train = train.drop(columns=["book"]).to_numpy()
y_train = train["book"].to_numpy()

X_test = test.drop(columns=["book"]).to_numpy()
y_test = test["book"].to_numpy()


# Calculate entropy
def entropy(y):
    counts = np.bincount(y.astype(int))
    probabilities = counts[counts > 0] / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


# Find the best feature using Gain Ratio
def best_split(X, y, features):
    parent_entropy = entropy(y)
    best_ratio = -1
    best_feature = None

    for feature in features:
        left = X[:, feature] == 0
        right = X[:, feature] == 1

        if not left.any() or not right.any():
            continue

        left_y = y[left]
        right_y = y[right]

        weighted_entropy = (
            len(left_y) / len(y) * entropy(left_y)
            + len(right_y) / len(y) * entropy(right_y)
        )

        information_gain = parent_entropy - weighted_entropy

        split_probability = len(left_y) / len(y)
        split_info = -(
            split_probability * np.log2(split_probability)
            + (1 - split_probability) * np.log2(1 - split_probability)
        )

        if split_info == 0:
            continue

        gain_ratio = information_gain / split_info

        if gain_ratio > best_ratio:
            best_ratio = gain_ratio
            best_feature = feature

    return best_feature


# Majority class
def majority_class(y):
    return Counter(y).most_common(1)[0][0]


# Build C4.5-style decision tree
def build_tree(X, y, features, depth=0, max_depth=8):

    majority = majority_class(y)

    if len(set(y)) == 1 or not features or depth >= max_depth:
        return {"class": majority}

    feature = best_split(X, y, features)

    if feature is None:
        return {"class": majority}

    left = X[:, feature] == 0
    right = X[:, feature] == 1

    remaining = [f for f in features if f != feature]

    return {
        "feature": feature,
        "left": build_tree(
            X[left], y[left], remaining, depth + 1, max_depth
        ),
        "right": build_tree(
            X[right], y[right], remaining, depth + 1, max_depth
        ),
        "class": majority
    }


# Predict class
def predict_one(row, tree):

    if "feature" not in tree:
        return tree["class"]

    feature = tree["feature"]

    if row[feature] == 0:
        return predict_one(row, tree["left"])

    return predict_one(row, tree["right"])


# Train the model
features = list(range(X_train.shape[1]))

tree = build_tree(X_train, y_train, features)

# Make predictions
predictions = np.array([
    predict_one(row, tree) for row in X_test
])

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("C4.5 DECISION TREE RESULTS")
print("--------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save predictions
results = pd.DataFrame({
    "Actual Book": y_test,
    "Predicted Book": predictions
})

results.to_csv(
    project_folder / "Results" / "c45_predictions.csv",
    index=False
)

print("\nC4.5 predictions saved successfully!")