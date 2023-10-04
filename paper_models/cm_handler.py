import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

def make_cm(y_actual=np.arange(5),
            y_pred=np.arange(5),
            name="Default Confusion Matrix Chart!",
            path=".",
            labels=[f"Value = {i}" for i in range(0, 5)],
            ):

    plt.cla()
    plt.clf()

    confusion_mtx = confusion_matrix(y_actual, y_pred)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx,
                                        display_labels=labels)
    cm_display.plot()
    ax = plt.gca()
    ax.set_xticklabels(labels, rotation=45)

    plt.title(name)
    plt.savefig(f"{path}/{name.replace(' ', '_')}.png")


if __name__ == "__main__":
    make_cm()
