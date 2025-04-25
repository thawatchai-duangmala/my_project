# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page settings
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

# Title
st.title("K-Means Clustering App with Mall Customer Dataset by Thawatchai Duangmala")

# Sidebar for clustering settings
st.sidebar.header("Configure Clustering")
num_clusters = st.sidebar.slider("Select number of Clusters", min_value=2, max_value=10, value=5)

# Upload CSV
data_file = st.sidebar.file_uploader("Upload Mall_Customers.csv", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)

    if "Annual Income (k$)" in df.columns and "Spending Score (1-100)" in df.columns:
        st.subheader("📋 Preview of Data")
        st.dataframe(df.head())

        # Select and scale features
        X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA to reduce to 2D for plotting
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Apply KMeans
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X_scaled)

        # Plotting
        colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(num_clusters):
            ax.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1],
                       c=[colors[i]], label=f"Cluster {i}", s=80)

        ax.set_title("Clusters (2D PCA Projection)", fontsize=14)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("❌ Required columns not found: 'Annual Income (k$)' and 'Spending Score (1-100)'")
else:
    st.info("📤 Please upload `Mall_Customers.csv` to get started.")
