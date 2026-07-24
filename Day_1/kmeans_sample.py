from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data (X, Y coordinates)
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3]
])

# Create K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Train the model
kmeans.fit(X)

# Cluster labels
labels = kmeans.labels_

# Cluster centers
centers = kmeans.cluster_centers_

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, s=100)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)
plt.title("K-Means Clustering")
plt.show()

print("Cluster Labels:", labels)
print("Cluster Centers:")
print(centers)