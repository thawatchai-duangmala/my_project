# app.py
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Streamlit page config
st.set_page_config(page_title="Customer Segmentation App", layout="centered")
st.title("Customer Segmentation with KMeans\nby Thawatchai Duangmala")

# Sidebar
st.sidebar.header("Upload Files & Configure")

# Upload dataset and model
data_file = st.sidebar.file_uploader("Upload Customer CSV", type=["csv"])
model_file = st.sidebar.file_uploader("Upload KMeans Model (.pkl)", type=["pkl"])

if data_file is not None and model_file is not None:
    # Load data
    df = pd.read_csv(data_file)
    
    # Show preview
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Check required columns
    if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

        # Load the model
        kmeans = pickle.load(model_file)

        # Predict clusters
        y_kmeans = kmeans.predict(X)

        # Define color palette
        n_clusters = kmeans.n_clusters
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        # Plot
        plt.figure(figsize=(8, 6))
        for i in range(n_clusters):
            plt.scatter(X.values[y_kmeans == i, 0], X.values[y_kmeans == i, 1],
                        c=[colors[i]], s=50, label=f"Cluster {i}")
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    s=300, c='red', label='Centroids', marker='X')
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        plt.title("Customer Segments")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")
else:
    st.info("Please upload both a dataset (.csv) and a KMeans model (.pkl).")
