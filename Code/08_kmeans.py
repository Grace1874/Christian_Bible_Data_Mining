
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

project_folder = Path(__file__).resolve().parent.parent

# Load cleaned Bible verses
data = pd.read_csv(
    project_folder / "Results" / "cleaned_bible_verses.csv"
)

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words="english"
)

X = vectorizer.fit_transform(
    data["cleaned_text"].fillna("")
)

# Apply K-Means clustering
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# Add cluster labels
data["Cluster"] = clusters

# Calculate silhouette score using a sample
sample_size = min(2000, X.shape[0])
sample_score = silhouette_score(
    X,
    clusters,
    sample_size=sample_size,
    random_state=42
)

# Display results
print("K-MEANS CLUSTERING RESULTS")
print("--------------------------")
print("Number of clusters:", 5)
print("Silhouette Score:", round(sample_score, 4))

print("\nNumber of verses in each cluster:")
print(data["Cluster"].value_counts().sort_index())

# Save results
data.to_csv(
    project_folder / "Results" / "kmeans_clusters.csv",
    index=False
)

print("\nK-Means results saved successfully!")