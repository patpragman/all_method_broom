"""
ChatGPT generated code summary

This Python script implements a logistic regression model using sci-kit learn for image classification. The
`get_best_model_logreg` function trains the logistic regression model on datasets of different image sizes and logs
the results using the WandB library. The classification report and confusion matrix are generated and saved for each
dataset.

Note: Ensure you have the required libraries and dataset paths configured before running this script.

"""


#  Logistic Regression Model using sci-kit learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
from skimage import color, io
import os
import pickle
from paper_models.cm_handler import make_cm
from pathlib import Path
HOME_DIRECTORY = Path.home()


def get_best_model_logreg(seed=42):
    # start up wandb!

    wandb.init(
        project="Elodea LogReg",
    )

    sizes = [224]
    #sizes.reverse()  # start with the smaller problem

    results_filename = 'results/logreg/log_reg_results.md'
    if os.path.isfile(results_filename):
        os.remove(results_filename)  # delete the old one, make a new one!

    with open(results_filename, "w") as results_file:
        results_file.write("# Results for Logistic Regression:\n")


    folder_paths = [f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{size}" for size in sizes]
    for size, dataset_path in zip(sizes, folder_paths):
        wandb.log({"msg": f"Logistic Regression Model for {size} x {size} images"})

        # variables to hold our data
        data = []
        Y = []

        classifier = LogisticRegression(class_weight='balanced', max_iter=3000)
        classes = os.listdir(dataset_path)

        excluded = ["pkl", ".DS_Store", "md", "png"]
        classes = [str(klass).split("/")[-1] for klass in Path(dataset_path).iterdir()
                   if klass.is_dir()]

        # create a mapping from the classes to each number class and demapping
        mapping = {n: i for i, n in enumerate(classes)}
        demapping = {i: n for i, n in enumerate(classes)}

        # now create an encoder
        encoder = lambda s: mapping[s]
        decoder = lambda i: demapping[i]  # I don't use this - so safe to delete, but in previous versions I did
        # I left it in for reference

        # now walk through and load the data in the containers we constructed above
        for root, dirs, files in os.walk(dataset_path):

            for file in files:
                if ".JPEG" in file.upper() or ".JPG" in file.upper() or ".PNG" in file.upper():
                    key = root.split("/")[-1]

                    if "cm" in file:
                        continue
                    else:
                        img = io.imread(f"{root}/{file}", as_gray=True)
                        arr = np.asarray(img).reshape(size * size, )  # reshape into an array
                        data.append(arr)

                        Y.append(encoder(key))  # simple one hot encoding

        y = np.array(Y)
        X = np.array(data)

        # now we've loaded all the X values into a single array
        # and all the Y values into another one, let's do a train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=seed)  # for consistency

        # now fit the classifier
        # fit the model with data
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        cr = classification_report(
            y_test, y_pred, target_names=[key for key in mapping.keys()], output_dict=True
        )

        wandb.log(cr)
        # save it
        with open(f"results/logreg/logreg_{size}.pkl", "wb") as file:
            pickle.dump(classifier, file)

        make_cm(
            y_actual=y_test, y_pred=y_pred, labels=[key for key in mapping.keys()],
            name=f"Logistic Regression Image Size {size}x{size}",
            path=f"results/logreg"
        )

        with open(results_filename, "a") as outfile:
            outfile.write(f"{size}x{size} images\n")
            outfile.write(classification_report(
                y_test, y_pred, target_names=[key for key in mapping.keys()]
            ))
            outfile.write("\n")

if __name__ == "__main__":
    get_best_model_logreg()
