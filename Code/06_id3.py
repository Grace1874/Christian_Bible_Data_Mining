
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# Locate project folder
project_folder = Path(__file__).resolve().parent.parent

# Load prepared datasets
train = pd.read_csv(project_folder / "Results" / "tree_train.csv")
test = pd.read_csv(project_folder / "Results" / "tree_test.csv")

X_train = train.drop(columns=["book"])
y_train = train["book"]

X_test = test.drop(columns=["book"])
y_test = test["book"]

# ID3 uses entropy and information gain
# One-hot binary word features are categorical inputs
id3 = DecisionTreeClassifier(
    criterion="entropy",
    splitter="best",
    random_state=42
)

# Train the ID3-style decision tree
id3.fit(X_train, y_train)

# Make predictions
predictions = id3.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("ID3 DECISION TREE RESULTS")
print("-------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save results
results = pd.DataFrame({
    "Actual Book": y_test,
    "Predicted Book": predictions
})

results.to_csv(
    project_folder / "Results" / "id3_predictions.csv",
    index=False
)

print("\nID3 predictions saved successfully!")
