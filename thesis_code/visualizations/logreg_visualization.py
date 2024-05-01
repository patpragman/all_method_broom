import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV


# Generate a simple binary classification dataset
X, y = make_classification(n_samples=100, n_features=1, n_classes=2, n_clusters_per_class=1, flip_y=0.03, n_informative=1, n_redundant=0, n_repeated=0)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Generate a range of values to predict
X_test = np.linspace(min(X), max(X), 300)

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Predict class labels for the original data
y_pred = model.predict(X)

# Identify misclassified points
misclassified = y != y_pred

# Define the decision threshold
threshold = 0.5

# Plotting
plt.scatter(X[misclassified], y[misclassified], color='orange', zorder=20, label='Misclassified')
plt.scatter(X[~misclassified], y[~misclassified], color='black', zorder=20, label='Correctly Classified')
plt.plot(X_test, probabilities, color='red', linewidth=3, label='Logistic regression curve')  # Logistic regression curve
plt.axhline(y=threshold, color='blue', linestyle='--', label='Decision Threshold')
plt.title('Logistic Regression with Misclassified Points')
plt.xlabel('Feature')
plt.ylabel('Probability')
plt.legend()

plt.savefig("logreg_visualization.png")
