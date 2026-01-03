import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common import prepare_data, show_images

# -------------------------------------------------

def recommend(query_index, embeddings, top_k=10):
    query_vec = embeddings[query_index].reshape(1, -1)

    # Cosine similarity (now meaningful due to normalization)
    scores = cosine_similarity(query_vec, embeddings)[0]

    top_indices = np.argsort(scores)[::-1][1:top_k + 1]
    return top_indices


def precision_at_k(query_index, retrieved_indices, labels):
    query_label = labels[query_index]
    relevant = sum(labels[idx] == query_label for idx in retrieved_indices)
    return relevant / len(retrieved_indices)

# -------------------------------------------------

if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)

    embeddings, images = joblib.load("output/embeddings.pkl")

    dataset = prepare_data()
    labels = [dataset[i][1] for i in range(len(dataset))]

    query_index = 42
    top_k = 10

    retrieved_indices = recommend(query_index, embeddings, top_k)
    precision = precision_at_k(query_index, retrieved_indices, labels)

    print(f"\nQuery Image Index: {query_index}")
    print(f"Precision@{top_k}: {precision:.2f}")

    display_images = [images[query_index]] + [images[i] for i in retrieved_indices]

    show_images(
        display_images,
        rows=3,
        cols=4,
        filename="output/recommendations.png"
    )

    print("Saved image: output/recommendations.png")
