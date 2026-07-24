import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# mock Mall Customers dataset
np.random.seed(42)
data = {
    'Annual_Income_k': np.random.randint(15, 130, 200),
    'Spending_Score_1_100': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)
X = df[['Annual_Income_k', 'Spending_Score_1_100']].values

# Feature Normalization (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means implementation with varying K values
plt.figure(figsize=(14, 4))
k_values = [2, 4, 6]

for idx, k in enumerate(k_values, 1):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    plt.subplot(1, 3, idx)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=150, label='Centroids')
    plt.title(f'K-Means Clustering (K = {k})')
    plt.xlabel('Income (Scaled)')
    plt.ylabel('Spending Score (Scaled)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()