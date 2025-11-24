# module used for data handling
import pandas as pd
import numpy as np
import json

# module used for model training
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# module used for hyperparameter tuning
from sklearn.metrics import silhouette_score

# constants
MAX_TOKENS = 8096
NUM_TOKENS_PER_REASON = 150
CW_BUDGET = MAX_TOKENS // NUM_TOKENS_PER_REASON

def get_diverse_examples(shap_values_fp: str):

    shap_values = pd.read_csv(shap_values_fp)

    scaler = StandardScaler()
    shap_values_scaled = scaler.fit_transform(shap_values)

    # finding best number of clusters
    N_sqrt = int(np.sqrt(shap_values.shape[0]))
    MAX_CLUSTERS = min(CW_BUDGET, N_sqrt)

    running_max_silhouette = -1
    best_n_clusters = 2
    for n_clusters in range(2, MAX_CLUSTERS + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(shap_values_scaled)

        silhouette_avg = silhouette_score(shap_values_scaled, 
                                          cluster_labels)

        if silhouette_avg > running_max_silhouette:
            running_max_silhouette = silhouette_avg
            best_n_clusters = n_clusters
    
    print(f"Found best number of clusters: k={best_n_clusters} with silhouette score: {running_max_silhouette}")

    # getting diverse indices
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(shap_values_scaled)

    diverse_indices = []
    for cluster_id in range(best_n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_center = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(shap_values_scaled[cluster_indices] - cluster_center, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        diverse_indices.append(closest_index)
    
    print(f"Chosen {len(diverse_indices)} diverse examples.")
    
    return diverse_indices
