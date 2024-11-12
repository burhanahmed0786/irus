# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Streamlit page configuration
st.title("Iris Dataset Exploration and KNN Classification")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["Data Overview", "Data Visualization", "Model Training"])

# Data Overview
if options == "Data Overview":
    st.header("Data Overview")
    st.write("### First few rows of the dataset")
    st.write(df.head())
    
    st.write("### Summary statistics")
    st.write(df.describe())
    
    st.write("### Check for missing values")
    st.write(df.isnull().sum())

# Data Visualization
elif options == "Data Visualization":
    st.header("Data Visualization")
    
    st.write("### Pairplot to visualize relationships between features")
    sns.pairplot(df, hue='species')
    st.pyplot(plt)
    
    st.write("### Correlation heatmap")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Model Training
elif options == "Model Training":
    st.header("K-Nearest Neighbors (KNN) Classification")
    
    # Separate features and target
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Select K for KNN
    k = st.slider("Select the number of neighbors (K)", 1, 15, 3)
    
    # Initialize and train the model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"### Accuracy of KNN model with K={k}: {accuracy * 100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    st.pyplot(plt)
    
    # Classification Report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

# Run the Streamlit app
# To run, open a terminal and type: streamlit run app.py
