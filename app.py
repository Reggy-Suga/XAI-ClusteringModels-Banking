import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess data
def preprocess_data(data):
    data = data.set_index("CUST_ID")
    data_numeric = data.select_dtypes(include=[np.number])  # Keep only numeric columns
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_numeric)
    return data_normalized, data_numeric.columns

# Load the KMeans model
@st.cache_resource
def load_kmeans_model():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Streamlit app
st.title('Bank Customer Segmentation App')

# Add logo to sidebar
st.sidebar.image('logo.png', use_column_width=True)

# Uploaded file
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    data_normalized, features_columns = preprocess_data(df)

    # Load the KMeans model
    kmeans_model = load_kmeans_model()

    # K-means clustering
    clusters = kmeans_model.predict(data_normalized)

    df['Cluster'] = clusters
    st.write("Data with cluster assignments:")
    st.write(df.astype({'CUST_ID': 'str'}).rename(columns={"CUST_ID": "Customer ID"}).replace({",": ""}, regex=True))



    # Graphical representation
    st.subheader("Graphical Representation")
    feature_x = st.selectbox("Select feature for X-axis:", features_columns)
    feature_y = st.selectbox("Select feature for Y-axis:", features_columns)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='Cluster', palette='viridis', ax=ax)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title("Clustering Result")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend to the right
    st.pyplot(fig)
