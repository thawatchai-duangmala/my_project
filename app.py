# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

# App title
st.title("K-Means Clustering App on Mall Customers Dataset")

# Sidebar for selecting number of clusters
st.sidebar.header("Configure Clustering")
num_clusters = st.sidebar.slider("Select number of Clusters", min_value=2, max_value=10, value=5)

# Load dataset (make sure this CSV is in the same directory)
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

# Extract the relevant features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Plot the clusters
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
for i in range(num_clusters):
    ax.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1], s=100, color=colors[i], label=f"Cluster {i}")

# Plot centroids
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           s=300, c='red', marker='X', label='Centroids')

ax.set_title("Customer Segments")
ax.set_xlabel("Annual Income (scaled)")
ax.set_ylabel("Spending Score (scaled)")
ax.legend()
st.pyplot(fig)
