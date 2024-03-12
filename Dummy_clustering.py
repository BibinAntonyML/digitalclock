#demo code
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate some dummy data
n_samples = 300
centers = 4
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.60, random_state=0)

# Visualize the dummy data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dummy Data')
plt.show()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Visualize the clustered data
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustered Data')
plt.show()
