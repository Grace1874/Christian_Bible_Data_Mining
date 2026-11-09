import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

project_folder = Path(__file__).resolve().parent.parent

train = pd.read_csv(project_folder / "Results" / "tree_train.csv")
test = pd.read_csv(project_folder / "Results" / "tree_test.csv")

X_train = train.drop(columns=["book"])
y_train = train["book"]

X_test = test.drop(columns=["book"])
y_test = test["book"]

# CART Decision Tree using Gini impurity
cart = DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    random_state=42
)

cart.fit(X_train, y_train)

predictions = cart.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("CART DECISION TREE RESULTS")
print("--------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

results = pd.DataFrame({
    "Actual Book": y_test,
    "Predicted Book": predictions
})

results.to_csv(
    project_folder / "Results" / "cart_predictions.csv",
    index=False
)

print("\nCART predictions saved successfully!")