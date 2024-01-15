import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)  # 2 standard deviations
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ell = Ellipse(xy=position, width=nsig * width, height=nsig * height, angle=angle, **kwargs)
        ax.add_patch(ell)

# Generate a simple dataset with 5 features
X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)

# Create and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Predict the cluster labels
labels = gmm.predict(X)

# For visualization, we'll only use the first two features
X_vis = X[:, :2]

# Plotting the results
plt.figure(figsize=(10, 8))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

# Plot the centroids and the standard deviation ellipses
for pos, covar in zip(gmm.means_[:, :2], gmm.covariances_[:, :2, :2]):
    draw_ellipse(pos, covar, alpha=0.2, color='red')

plt.title('Gaussian Mixture Model (GMM) Clustering with Centroid Ellipses (Visualized with 2 of 5 Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')


plt.savefig("gmm_visualization.png")
