import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

# Generate a simple binary classification dataset with 5 features
X, y = make_classification(n_samples=100, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=42)

# Create and train the SVM model
model = svm.SVC(kernel='rbf', C=1)
model.fit(X, y)

# Predictions on the dataset
y_pred = model.predict(X)

# Identify misclassified points
misclassified = y != y_pred

# For visualization, we'll only use the first two features
X_vis = X[:, :2]

# Create a mesh grid to plot the decision boundaries for visualization
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Use the model to make predictions just for visualization
# We create a temporary dataset combining the mesh grid points with zeros for the missing features
temp_dataset = np.c_[xx.ravel(), yy.ravel(), np.zeros((len(xx.ravel()), 3))]
Z = model.predict(temp_dataset)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8)

# Plot the training points
plt.scatter(X_vis[misclassified, 0], X_vis[misclassified, 1], color='red', label='Misclassified', edgecolor='k')
plt.scatter(X_vis[~misclassified, 0], X_vis[~misclassified, 1], color='blue', label='Correctly Classified', edgecolor='k')

plt.title('SVM Classification with Misclassified Points (Visualized with 2 of 5 Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


plt.savefig("svm_visualization.png")
