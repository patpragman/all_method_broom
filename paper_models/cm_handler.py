"""
ChatGPT generated code summary

This Python script defines a function `make_cm` for creating a confusion matrix heatmap using the Seaborn library.
The script also includes a simple example of using this function with default values in the `__main__` block. The
confusion matrix is saved as a PNG file. The script demonstrates the visualization of a confusion matrix and provides
flexibility for customization. """


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV

def make_cm(y_actual=np.arange(5),
            y_pred=np.arange(5),
            name="Default Confusion Matrix Chart!",
            path=".",
            labels=[f"Value = {i}" for i in range(0, 5)],
            ):

    plt.cla()
    plt.clf()

    conf_matrix = confusion_matrix(y_actual, y_pred)

    # Create a heatmap to visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {name} data')
    plt.savefig(f"{path}/{name.replace(' ', '_')}.png")


if __name__ == "__main__":
    make_cm()
