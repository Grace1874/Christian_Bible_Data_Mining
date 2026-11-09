
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

project_folder = Path(__file__).resolve().parent.parent

# Load cleaned Bible verses
data = pd.read_csv(
    project_folder / "Results" / "cleaned_bible_verses.csv"
)

# Use a manageable sample for hierarchical clustering
data = data.sample(
    n=min(2000, len(data)),
    random_state=42
).reset_index(drop=True)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=300,
    stop_words="english"
)

X = vectorizer.fit_transform(
    data["cleaned_text"].fillna("")
)

# Apply hierarchical clustering
model = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

clusters = model.fit_predict(X.toarray())

# Add cluster labels
data["Cluster"] = clusters

# Calculate silhouette score
score = silhouette_score(X, clusters)

print("HIERARCHICAL CLUSTERING RESULTS")
print("--------------------------------")
print("Number of clusters:", 5)
print("Silhouette Score:", round(score, 4))

print("\nNumber of verses in each cluster:")
print(data["Cluster"].value_counts().sort_index())

# Save results
data.to_csv(
    project_folder / "Results" / "hierarchical_clusters.csv",
    index=False
)

print("\nHierarchical clustering results saved successfully!")
