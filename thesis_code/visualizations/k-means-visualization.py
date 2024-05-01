import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV





import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate a simple dataset with 5 features
X, _ = make_blobs(n_samples=150, n_features=20, centers=4, random_state=42)

# Create and fit the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predict the cluster labels
labels = kmeans.predict(X)

# For visualization, we'll only use the first two features
X_vis = X[:, :2]

# Plotting the results
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, s=50, cmap='viridis')

# Plot the centroids
centers = kmeans.cluster_centers_[:, :2]
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.savefig("kmeans_visualization.png")
