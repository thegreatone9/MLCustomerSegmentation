"""
Clustering Algorithm Comparison — Mall Customer Segmentation
=============================================================
Compares: K-Means, Hierarchical (Agglomerative), DBSCAN, Gaussian Mixture Models
Run: python research/algorithm_comparison.py
Outputs saved to: research/figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

os.makedirs("research/figures", exist_ok=True)
SAVE_DIR = "research/figures"


def load_and_prepare():
    """Load data and apply StandardScaler."""
    df = pd.read_csv("data/Mall_Customers.csv")
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled


# ================================================================
# 1. ELBOW METHOD — Finding optimal K for KMeans
# ================================================================
def plot_elbow_method(X_scaled):
    """Plot the elbow curve to determine optimal number of clusters."""
    inertias = []
    silhouettes = []
    K_range = range(2, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow curve
    ax1.plot(K_range, inertias, "o-", color="#667eea", linewidth=2, markersize=8)
    ax1.axvline(x=5, color="#e74c3c", linestyle="--", alpha=0.7, label="Optimal k=5")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=12)
    ax1.set_title("Elbow Method", fontweight="bold", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_xticks(list(K_range))

    # Silhouette scores
    colors = ["#e74c3c" if k == 5 else "#667eea" for k in K_range]
    ax2.bar(K_range, silhouettes, color=colors, alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score by k", fontweight="bold", fontsize=14)
    ax2.set_xticks(list(K_range))
    for k, s in zip(K_range, silhouettes):
        ax2.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/08_elbow_method.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 08_elbow_method.png")


# ================================================================
# 2. HIERARCHICAL CLUSTERING — Dendrogram
# ================================================================
def plot_dendrogram(X_scaled):
    """Plot dendrogram for hierarchical clustering."""
    fig, ax = plt.subplots(figsize=(14, 6))
    Z = linkage(X_scaled, method="ward")
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90,
               leaf_font_size=10, ax=ax, color_threshold=7,
               above_threshold_color="#667eea")
    ax.axhline(y=7, color="#e74c3c", linestyle="--", linewidth=2, label="Cut at 5 clusters")
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/09_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 09_dendrogram.png")


# ================================================================
# 3. FOUR-ALGORITHM COMPARISON
# ================================================================
def compare_algorithms(X_scaled, X_raw):
    """Run all 4 algorithms and compare side by side."""
    algorithms = {}

    # K-Means (k=5)
    km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
    algorithms["K-Means (k=5)"] = km.fit_predict(X_scaled)

    # Agglomerative Hierarchical (k=5)
    agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
    algorithms["Hierarchical (Ward)"] = agg.fit_predict(X_scaled)

    # DBSCAN (tune eps for this dataset)
    db = DBSCAN(eps=0.45, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    algorithms["DBSCAN"] = db_labels

    # Gaussian Mixture Model (k=5)
    gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
    algorithms["GMM (k=5)"] = gmm.fit_predict(X_scaled)

    # --- Plot all 4 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    palette = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#3498db",
               "#1abc9c", "#e67e22", "#34495e", "#95a5a6"]

    for idx, (name, labels) in enumerate(algorithms.items()):
        ax = axes[idx]
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            mask = labels == label
            if label == -1:  # DBSCAN noise
                ax.scatter(X_raw[mask, 0], X_raw[mask, 1], c="gray", marker="x",
                           s=30, alpha=0.5, label="Noise")
            else:
                color = palette[label % len(palette)]
                ax.scatter(X_raw[mask, 0], X_raw[mask, 1], c=color,
                           s=50, alpha=0.7, edgecolors="white", linewidth=0.5,
                           label=f"Cluster {label}")

        ax.set_xlabel("Annual Income (k$)", fontsize=11)
        ax.set_ylabel("Spending Score (1-100)", fontsize=11)
        ax.set_title(name, fontweight="bold", fontsize=13)
        ax.legend(fontsize=8, loc="best")

        # Add metrics
        valid = labels != -1
        if len(set(labels[valid])) >= 2:
            sil = silhouette_score(X_scaled[valid], labels[valid])
            db_score = davies_bouldin_score(X_scaled[valid], labels[valid])
            ch = calinski_harabasz_score(X_scaled[valid], labels[valid])
            n_clusters = len(set(labels[valid]))
            metrics_text = f"Clusters: {n_clusters}\nSilhouette: {sil:.3f}\nDBI: {db_score:.3f}\nCH: {ch:.1f}"
            if -1 in labels:
                n_noise = (labels == -1).sum()
                metrics_text += f"\nNoise: {n_noise}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4",
                    facecolor="white", alpha=0.8))

    plt.suptitle("Clustering Algorithm Comparison", fontweight="bold", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/10_algorithm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 10_algorithm_comparison.png")

    return algorithms


# ================================================================
# 4. METRICS COMPARISON TABLE
# ================================================================
def print_metrics_table(X_scaled, algorithms):
    """Print a comparison table of all metrics."""
    print("\n" + "=" * 80)
    print("CLUSTERING ALGORITHM COMPARISON — METRICS")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Clusters':>8} {'Silhouette':>12} {'Davies-Bouldin':>15} {'Calinski-H':>12} {'Noise':>7}")
    print("-" * 80)

    results = []
    for name, labels in algorithms.items():
        valid = labels != -1
        n_clusters = len(set(labels[valid]))
        n_noise = (labels == -1).sum()

        if n_clusters >= 2:
            sil = silhouette_score(X_scaled[valid], labels[valid])
            dbi = davies_bouldin_score(X_scaled[valid], labels[valid])
            ch = calinski_harabasz_score(X_scaled[valid], labels[valid])
        else:
            sil = dbi = ch = float("nan")

        results.append({
            "Algorithm": name, "Clusters": n_clusters,
            "Silhouette": sil, "Davies-Bouldin": dbi,
            "Calinski-Harabasz": ch, "Noise": n_noise
        })
        print(f"{name:<25} {n_clusters:>8} {sil:>12.4f} {dbi:>15.4f} {ch:>12.1f} {n_noise:>7}")

    print("-" * 80)
    print("\nInterpretation:")
    print("  Silhouette Score:    Higher is better (range: -1 to 1)")
    print("  Davies-Bouldin:      Lower is better (0 = perfect)")
    print("  Calinski-Harabasz:   Higher is better")
    print("  Noise:               Points not assigned to any cluster (DBSCAN only)")

    return results


# ================================================================
# 5. CLUSTER STABILITY ANALYSIS
# ================================================================
def cluster_stability_analysis(X_scaled, n_runs=20):
    """Test cluster stability by running KMeans with different random seeds
    and measuring consistency of assignments."""
    print("\n" + "=" * 60)
    print("CLUSTER STABILITY ANALYSIS")
    print("=" * 60)

    n_samples = X_scaled.shape[0]
    all_labels = np.zeros((n_runs, n_samples), dtype=int)
    all_silhouettes = []
    all_inertias = []

    for i in range(n_runs):
        km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=i * 7)
        labels = km.fit_predict(X_scaled)
        all_labels[i] = labels
        all_silhouettes.append(silhouette_score(X_scaled, labels))
        all_inertias.append(km.inertia_)

    # Agreement matrix: how often are two points in the same cluster?
    agreement = np.zeros((n_samples, n_samples))
    for labels in all_labels:
        for i in range(n_samples):
            agreement[i] = agreement[i] + (labels == labels[i])
    agreement /= n_runs

    avg_agreement = (agreement.sum() - n_samples) / (n_samples * (n_samples - 1))

    print(f"\nRuns: {n_runs} (different random seeds)")
    print(f"Silhouette Score:  {np.mean(all_silhouettes):.4f} +/- {np.std(all_silhouettes):.4f}")
    print(f"Inertia:           {np.mean(all_inertias):.2f} +/- {np.std(all_inertias):.2f}")
    print(f"Pairwise Agreement: {avg_agreement:.4f} (1.0 = perfectly stable)")

    # Plot stability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette across runs
    ax1.bar(range(n_runs), all_silhouettes, color="#667eea", alpha=0.8, edgecolor="white")
    ax1.axhline(y=np.mean(all_silhouettes), color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(all_silhouettes):.4f}")
    ax1.fill_between(range(n_runs),
                     np.mean(all_silhouettes) - np.std(all_silhouettes),
                     np.mean(all_silhouettes) + np.std(all_silhouettes),
                     color="#e74c3c", alpha=0.1)
    ax1.set_xlabel("Run", fontsize=12)
    ax1.set_ylabel("Silhouette Score", fontsize=12)
    ax1.set_title("Cluster Stability — Silhouette Score", fontweight="bold", fontsize=13)
    ax1.legend(fontsize=11)

    # Co-occurrence matrix
    im = ax2.imshow(agreement, cmap="YlOrRd", aspect="auto")
    ax2.set_title("Pairwise Co-Clustering Agreement", fontweight="bold", fontsize=13)
    ax2.set_xlabel("Customer Index", fontsize=12)
    ax2.set_ylabel("Customer Index", fontsize=12)
    plt.colorbar(im, ax=ax2, label="Agreement Rate")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/11_cluster_stability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 11_cluster_stability.png")

    if np.std(all_silhouettes) < 0.01:
        print("\nVerdict: Clusters are HIGHLY STABLE across different initializations.")
    elif np.std(all_silhouettes) < 0.05:
        print("\nVerdict: Clusters are MODERATELY STABLE.")
    else:
        print("\nVerdict: Clusters show INSTABILITY — consider different k or algorithm.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   CLUSTERING ALGORITHM COMPARISON")
    print("=" * 60 + "\n")

    df, X_raw, X_scaled = load_and_prepare()
    print(f"Data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features (scaled)\n")

    plot_elbow_method(X_scaled)
    plot_dendrogram(X_scaled)
    algorithms = compare_algorithms(X_scaled, X_raw)
    print_metrics_table(X_scaled, algorithms)
    cluster_stability_analysis(X_scaled)

    print(f"\nAll figures saved to {SAVE_DIR}/")
    print("Comparison complete.")
