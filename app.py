import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_tracks.csv")
    return df

# Preprocess and fit similarity matrix
@st.cache_resource
def preprocess(df):
    features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    sim_matrix = cosine_similarity(scaled)
    return sim_matrix, df

# Recommend songs
def recommend(song_name, df, sim_matrix, n=5):
    if song_name not in df['name'].values:
        return []

    idx = df[df['name'] == song_name].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recommended = [df.iloc[i[0]]['name'] for i in scores]
    return recommended

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
