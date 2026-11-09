
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# 1. Locate and load the cleaned dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "Results" / "cleaned_bible_verses.csv"

df = pd.read_csv(file_path)

# 2. Select verse text and Bible book number
df = df[["cleaned_text", "b"]].dropna()

X_text = df["cleaned_text"]
y = df["b"]

# 3. Convert text into binary word features
vectorizer = CountVectorizer(
    binary=True,
    max_features=80,
    min_df=10,
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Save the prepared datasets
feature_names = vectorizer.get_feature_names_out()

train_df = pd.DataFrame(
    X_train.toarray(),
    columns=feature_names
)
train_df["book"] = y_train.reset_index(drop=True)

test_df = pd.DataFrame(
    X_test.toarray(),
    columns=feature_names
)
test_df["book"] = y_test.reset_index(drop=True)

train_df.to_csv(
    project_folder / "Results" / "tree_train.csv",
    index=False
)

test_df.to_csv(
    project_folder / "Results" / "tree_test.csv",
    index=False
)

print("DECISION TREE DATA PREPARATION COMPLETED!")
print("Total verses:", len(df))
print("Number of word features:", X.shape[1])
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Number of Bible book classes:", y.nunique())
print("\nFiles saved successfully!")
