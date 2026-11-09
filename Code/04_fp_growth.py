
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1. Locate and load the cleaned dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "Results" / "cleaned_bible_verses.csv"

df = pd.read_csv(file_path)

# 2. Remove common English stop words
stop_words = set(ENGLISH_STOP_WORDS)

def tokenize(text):
    words = text.split()
    return list(set(
        word for word in words
        if word not in stop_words and len(word) > 2
    ))

transactions = df["cleaned_text"].fillna("").apply(tokenize).tolist()

# 3. Select the 150 most frequent words
word_counts = Counter(
    word for transaction in transactions for word in transaction
)

selected_words = {
    word for word, count in word_counts.most_common(150)
}

transactions = [
    [word for word in transaction if word in selected_words]
    for transaction in transactions
]

transactions = [t for t in transactions if t]

print("Total transactions:", len(transactions))

# 4. Convert transactions into one-hot encoded format
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(
    transactions, sparse=True
)

basket = pd.DataFrame.sparse.from_spmatrix(
    encoded_array,
    columns=encoder.columns_
)

# 5. Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(
    basket,
    min_support=0.005,
    use_colnames=True,
    max_len=3
)

print("\nFP-GROWTH FREQUENT ITEMSETS")
print(frequent_itemsets.head(20))

# 6. Generate association rules
if len(frequent_itemsets) > 0:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.5
    )

    rules = rules.sort_values(
        by="lift", ascending=False
    )

    print("\nTOP FP-GROWTH ASSOCIATION RULES")
    print(rules.head(20))

    # 7. Save results
    frequent_itemsets.to_csv(
        project_folder / "Results" / "fp_growth_frequent_itemsets.csv",
        index=False
    )

    rules.to_csv(
        project_folder / "Results" / "fp_growth_association_rules.csv",
        index=False
    )

    print("\nFP-Growth results saved successfully!")

else:
    print("No frequent itemsets found.")
