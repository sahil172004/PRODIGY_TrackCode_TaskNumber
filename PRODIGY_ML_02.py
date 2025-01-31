import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset safely
file_path = "your_dataset.csv"  # Update this with your actual dataset path
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please check the path.")

df = pd.read_csv(file_path)

# Selecting relevant numerical features (Modify column names as per your dataset)
features = ["Total_Spend", "Frequency", "Avg_Spend_Per_Visit"]  # Example features

# Ensure all selected features exist in the dataset
missing_features = [col for col in features if col not in df.columns]
if missing_features:
    raise KeyError(f"Missing columns in dataset: {missing_features}")

# Handle missing values (Fill with median instead of dropping for better data retention)
df[features] = df[features].fillna(df[features].median())

# Standardizing the data for better clustering
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Using the Elbow Method to find optimal clusters
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='b')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method to Determine Optimal K")
plt.xticks(K_range)
plt.show()

# Applying K-Means Clustering with optimal K (Assume K=3 from Elbow Method)
optimal_k = 3  # Change based on Elbow method output
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Visualizing Clusters (Using first two features)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df["Cluster"], palette="viridis", s=100, edgecolor="black")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Customer Clusters")
plt.legend(title="Cluster")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Display sample data with cluster assignments
print(df.head())
