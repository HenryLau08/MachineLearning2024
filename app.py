import streamlit as st
import pandas as pd
import librosa
import librosa.display
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    data_path = "./labeled_data.csv"
    df = pd.read_csv(data_path)
    return df


def recommend_similar(df, audio_file, top_n=5):
    # Extract features and normalize
    features = df.drop(columns=["Unnamed: 0", "genre"])
    features_normalized = normalize(features, axis=0)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(features_normalized)

    # Find the index of the given audio file
    target_index = df[df["Unnamed: 0"] == audio_file].index[0]

    similar_indices = cosine_sim[target_index].argsort()[-(top_n + 1) : -1][::-1]

    similar_files = df.iloc[similar_indices][["Unnamed: 0", "genre"]]
    similarity_scores = cosine_sim[target_index][similar_indices]

    similar_files["similarity_score"] = similarity_scores

    return similar_files


# Streamlit app
def main():
    st.title("Dun Dun Dunnnnnn")

    df = load_data()

    audio_file = st.selectbox("Select an audio file:", df["Unnamed: 0"].unique())

    st.subheader(f"Playing Selected Audio: {audio_file}")
    selected_audio_path = f"Portfolio_3/labeled/{audio_file}"
    st.audio(selected_audio_path)

    if st.button("Get Recommendations"):
        recommendations = recommend_similar(df, audio_file)

        st.subheader(f"Recommendations for {audio_file}:")
        for i, row in recommendations.iterrows():
            st.write(
                f"{i+1}. {row['Unnamed: 0']} (Genre: {row['genre']}, Similarity: {row['similarity_score']:.2f})"
            )

            audio_path = f"Portfolio_3/labeled/{row['Unnamed: 0']}"
            st.audio(audio_path)


if __name__ == "__main__":
    main()
