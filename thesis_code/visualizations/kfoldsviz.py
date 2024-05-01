import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Number of samples in your dataset
num_samples = 100
# Number of folds
k = 5

# Generate a sample dataset
data = np.arange(num_samples)

# This will store the indices for each fold
fold_indices = np.array_split(data, k)

# Set up the plot
plt.figure(figsize=(12, 8))

for i, fold in enumerate(fold_indices):
    # The rest of the data, excluding the current fold, will be training data
    training_data = np.delete(data, np.arange(fold[0], fold[-1] + 1))

    # Plot training data
    sns.scatterplot(x=training_data, y=np.repeat(i, len(training_data)), s=100, color="skyblue",
                    label="Training Data" if i == 0 else "")

    # Plot validation data
    sns.scatterplot(x=fold, y=np.repeat(i, len(fold)), s=100, color="salmon", label="Validation Data" if i == 0 else "")

# Enhance plot
plt.yticks(ticks=np.arange(k), labels=[f"Fold {i + 1}" for i in range(k)])
plt.xlabel("Sample index")
plt.ylabel("Fold")
plt.title("Visualization of K-Folds Cross-Validation")
plt.legend()
plt.tight_layout()
plt.savefig("kfoldsviz.png")
