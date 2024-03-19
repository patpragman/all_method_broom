import os
from statistics import stdev, mean
import pickle
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV

import matplotlib.pyplot as plt
import pandas as pd

f1s = {}
accuracies = {}

for score_file in os.listdir("scores"):
    name = score_file.split("_")[0]
    with open(f'scores/{score_file}', "rb") as pickle_file:
        data = pickle.load(pickle_file)
        f1s[name] = data['f1_scores']
        accuracies[name] = data['accuracies']

dfs = [(pd.DataFrame(accuracies), "Accuracy"),
       (pd.DataFrame(f1s), "Macro $F_1$ Score")]
for df, title in dfs:
    df.boxplot()
    plt.ylabel(title)
    plt.title('Box Plot of K-Folds Cross Validation Various Models')
    plt.savefig(f'boxy_big_{title}_renamed_patnet.png')
    plt.clf()
    plt.cla()