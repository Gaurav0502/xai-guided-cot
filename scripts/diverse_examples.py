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
from scripts.constants import MAX_CLUSTERS_BUDGET

# module used for typing
from typing import List

# selects diverse
# decision-making examples
def get_diverse_examples(shap_values_fp: str) -> List[int]:
    """
    Selects a set of diverse decision-making examples using clustering on SHAP values.

    Args:
        shap_values_fp (str): File path to the CSV containing SHAP values with an 'idx' column.

    Returns:
        list: List of indices corresponding to medoids of each cluster.
    """

    # loading SHAP values
    shap_values = pd.read_csv(shap_values_fp)

    try:
        # map indices 
        # to original dataset
        shap_values.set_index("idx", inplace=True)
    except KeyError:
        raise KeyError(
            "SHAP values CSV must contain an 'idx' column "
            "for mapping to original indices."
        )
    except Exception:
        raise

    # scaling SHAP values
    scaler = StandardScaler()
    shap_values_scaled = scaler.fit_transform(shap_values)

    # upper bound on number
    # of clusters to try
    N_sqrt = int(np.sqrt(shap_values.shape[0]))
    MAX_CLUSTERS = min(MAX_CLUSTERS_BUDGET, N_sqrt)

    # hyperparameter tuning
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
    
    print(f"[DIVERSE EXAMPLES] Found best number of clusters: k={best_n_clusters} with silhouette score: {running_max_silhouette}")

    # getting diverse indices
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(shap_values_scaled)

    # locating the medoids
    diverse_indices = []
    for cluster_id in range(best_n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_center = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(shap_values_scaled[cluster_indices] - cluster_center, axis=1)
        closest_pos = cluster_indices[np.argmin(distances)]
        closest_index = shap_values.index[closest_pos]
        diverse_indices.append(closest_index)
    
    print(f"[DIVERSE EXAMPLES] Chosen {len(diverse_indices)} diverse examples.")
    
    return diverse_indices
