import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for compatibility
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Generate 4 blobs of data
X, y = make_blobs(n_samples=100, centers=4, n_features=2)

# Filter out one of the blobs (e.g., the blob with label '3')
mask = y != 3
X_train = X[mask]
y_train = y[mask]

# Train SVM on the 3 selected blobs
clf = SVC(kernel='poly').fit(X_train, y_train)

# Predict the entire dataset
y_pred = clf.predict(X)

# Identify mislabeled points
mislabeled = y != y_pred

# Create a mesh to plot decision boundaries
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict and plot decision boundaries
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot correctly labeled points
plt.scatter(X[~mislabeled, 0], X[~mislabeled, 1], c="blue", edgecolors='k', label='Correctly Labeled')

# Plot mislabeled points in red
plt.scatter(X[mislabeled, 0], X[mislabeled, 1], color='red', edgecolors='k', label='Mislabeled')

plt.title("SVM Decision Boundary with 4 Blobs (1 Missed)")
plt.legend()

plt.savefig("svm_visualization.png")
