import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans

st.title("Customer Segmentation with KMeans")

# Upload CSV data
csv_file = st.file_uploader("Upload Customer Dataset (CSV)", type=['csv'])
model_file = st.file_uploader("Upload Trained KMeans Model (.pkl)", type=["pkl"])

if csv_file is not None and model_file is not None:
    # Load data
    data = pd.read_csv(csv_file)
    st.write("Preview of Dataset:", data.head())

    # Select features
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Load model
    kmeans = pickle.load(model_file)

    # Predict
    y_kmeans = kmeans.predict(X)

    # Plot clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_kmeans, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Customer Segments')
    ax.legend()
    st.pyplot(fig)
