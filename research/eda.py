"""
Exploratory Data Analysis — Mall Customer Segmentation
======================================================
Run: python research/eda.py
Outputs saved to: research/figures/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Setup
os.makedirs("research/figures", exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
SAVE_DIR = "research/figures"


def load_data():
    """Load the Mall Customers dataset."""
    df = pd.read_csv("data/Mall_Customers.csv")
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    return df


def plot_distributions(df):
    """Plot distribution of each numerical feature."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    colors = ["#667eea", "#764ba2", "#f093fb"]

    for ax, feat, color in zip(axes, features, colors):
        ax.hist(df[feat], bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(df[feat].mean(), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {df[feat].mean():.1f}")
        ax.axvline(df[feat].median(), color="#2ecc71", linestyle="--", linewidth=2, label=f"Median: {df[feat].median():.1f}")
        ax.set_title(f"Distribution of {feat}", fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/01_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_distributions.png")


def plot_gender_analysis(df):
    """Analyze gender distribution and differences."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gender count
    gender_counts = df["Gender"].value_counts()
    colors = ["#667eea", "#f093fb"]
    axes[0].pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"fontweight": "bold"})
    axes[0].set_title("Gender Distribution", fontweight="bold")

    # Income by gender
    for i, gender in enumerate(["Male", "Female"]):
        subset = df[df["Gender"] == gender]
        axes[1].hist(subset["Annual Income (k$)"], bins=15, alpha=0.6,
                     color=colors[i], label=gender, edgecolor="white")
    axes[1].set_title("Income Distribution by Gender", fontweight="bold")
    axes[1].set_xlabel("Annual Income (k$)")
    axes[1].legend()

    # Spending by gender
    for i, gender in enumerate(["Male", "Female"]):
        subset = df[df["Gender"] == gender]
        axes[2].hist(subset["Spending Score (1-100)"], bins=15, alpha=0.6,
                     color=colors[i], label=gender, edgecolor="white")
    axes[2].set_title("Spending Score by Gender", fontweight="bold")
    axes[2].set_xlabel("Spending Score (1-100)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/02_gender_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_gender_analysis.png")


def plot_correlation_matrix(df):
    """Plot correlation heatmap of numerical features."""
    numerical = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    corr = numerical.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                mask=mask, square=True, linewidths=2, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/03_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 03_correlation_matrix.png")


def plot_pairplot(df):
    """Create pairwise scatter plots colored by gender."""
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
    g = sns.pairplot(df[features], hue="Gender", palette=["#667eea", "#f093fb"],
                     diag_kind="kde", plot_kws={"alpha": 0.6, "s": 40})
    g.fig.suptitle("Pairwise Feature Relationships", fontweight="bold", y=1.02)

    plt.savefig(f"{SAVE_DIR}/04_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_pairplot.png")


def plot_scatter_features(df):
    """Scatter plot of the two features used for clustering."""
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
                         c=df["Age"], cmap="viridis", s=60, alpha=0.7, edgecolors="white", linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Age", fontweight="bold")
    ax.set_xlabel("Annual Income (k$)", fontsize=12)
    ax.set_ylabel("Spending Score (1-100)", fontsize=12)
    ax.set_title("Income vs Spending Score (colored by Age)", fontweight="bold", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/05_income_vs_spending.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_income_vs_spending.png")


def plot_boxplots(df):
    """Box plots showing feature distributions and outliers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    colors = ["#667eea", "#764ba2", "#f093fb"]

    for ax, feat, color in zip(axes, features, colors):
        bp = ax.boxplot(df[feat], patch_artist=True, widths=0.6,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker="o", markerfacecolor="#e74c3c", markersize=8))
        ax.set_title(f"{feat}", fontweight="bold")
        ax.set_ylabel("Value")

        # Add stats
        q1, q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
        iqr = q3 - q1
        ax.text(1.3, df[feat].median(), f"Median: {df[feat].median():.0f}", fontsize=9, va="center")
        ax.text(1.3, q1, f"Q1: {q1:.0f}", fontsize=9, va="center")
        ax.text(1.3, q3, f"Q3: {q3:.0f}", fontsize=9, va="center")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/06_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 06_boxplots.png")


def plot_age_segments(df):
    """Analyze spending patterns across age groups."""
    bins = [0, 25, 35, 45, 55, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "55+"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Average income by age group
    age_income = df.groupby("Age Group", observed=True)["Annual Income (k$)"].mean()
    axes[0].bar(age_income.index, age_income.values, color="#667eea", alpha=0.8, edgecolor="white")
    axes[0].set_title("Average Income by Age Group", fontweight="bold")
    axes[0].set_ylabel("Annual Income (k$)")
    for i, v in enumerate(age_income.values):
        axes[0].text(i, v + 1, f"${v:.0f}k", ha="center", fontweight="bold")

    # Average spending by age group
    age_spending = df.groupby("Age Group", observed=True)["Spending Score (1-100)"].mean()
    axes[1].bar(age_spending.index, age_spending.values, color="#f093fb", alpha=0.8, edgecolor="white")
    axes[1].set_title("Average Spending Score by Age Group", fontweight="bold")
    axes[1].set_ylabel("Spending Score")
    for i, v in enumerate(age_spending.values):
        axes[1].text(i, v + 1, f"{v:.0f}", ha="center", fontweight="bold")

    df.drop("Age Group", axis=1, inplace=True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/07_age_segments.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07_age_segments.png")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   CUSTOMER SEGMENTATION — Exploratory Data Analysis")
    print("=" * 60 + "\n")

    df = load_data()
    print("\nGenerating visualizations...\n")

    plot_distributions(df)
    plot_gender_analysis(df)
    plot_correlation_matrix(df)
    plot_pairplot(df)
    plot_scatter_features(df)
    plot_boxplots(df)
    plot_age_segments(df)

    print(f"\nAll figures saved to {SAVE_DIR}/")
    print("EDA complete.")
