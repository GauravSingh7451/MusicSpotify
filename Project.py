import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Select numerical audio features for clustering
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- K-Means Clustering ----
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

# Plot silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(K_range, sil_scores, marker='o')
plt.title('Silhouette Score for K-Means')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Choose best K
best_k = K_range[np.argmax(sil_scores)]
print(f"Best K (highest Silhouette Score): {best_k}")

# Final KMeans model
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_db_index = davies_bouldin_score(X_scaled, kmeans_labels)
print(f"KMeans DB Index: {kmeans_db_index:.4f}")

# ---- DBSCAN Clustering ----
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Remove noise points (-1) for evaluation
mask = dbscan_labels != -1
if len(set(dbscan_labels[mask])) > 1:
    dbscan_sil = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_db_index = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
else:
    dbscan_sil = dbscan_db_index = None

print(f"DBSCAN Silhouette Score: {dbscan_sil if dbscan_sil else 'Not enough clusters'}")
print(f"DBSCAN DB Index: {dbscan_db_index if dbscan_db_index else 'Not enough clusters'}")

# ---- Add cluster labels to dataset ----
df_clustered = df.copy()
df_clustered['KMeans_Cluster'] = kmeans_labels
df_clustered['DBSCAN_Cluster'] = dbscan_labels

# Example: Recommend similar songs to a given track using KMeans
def recommend_similar(track_name, df, features, cluster_col='KMeans_Cluster', top_n=5):
    track = df[df['track_name'].str.lower() == track_name.lower()]
    if track.empty:
        return "Track not found."
    
    cluster = track[cluster_col].values[0]
    cluster_tracks = df[(df[cluster_col] == cluster) & (df['track_name'].str.lower() != track_name.lower())]
    return cluster_tracks.sample(n=min(top_n, len(cluster_tracks)))

# Example usage
print("\nSimilar songs to 'Blinding Lights':")
print(recommend_similar('Blinding Lights', df_clustered, features))

# Streamlit UI
def main():
    st.title("🎧 Spotify Music Recommender")
    df = load_data()
    sim_matrix, df = preprocess(df)

    song_list = df['name'].drop_duplicates().sort_values().tolist()
    song_choice = st.selectbox("Choose a Song", song_list)

    if st.button("Recommend"):
        results = recommend(song_choice, df, sim_matrix)
        if results:
            st.write("### Recommended Songs:")
            for r in results:
                st.write(f"- {r}")
        else:
            st.warning("No recommendations found.")

if __name__ == "__main__":
    main()

