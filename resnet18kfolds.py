import os
import pickle
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
acc = dfs[0][0]
f1 = dfs[1][0]

rn18_acc = acc['ResNet-18']
rn18_f1 = f1['ResNet-18']

rn18_f1.plot.box()
x_coord = 1
y_coord = 0.780
plt.plot(x_coord, y_coord, 'ro')
plt.title('K-Folds cross validation compared to previous hyperparameter sweep')
plt.annotate('Best $F_1$ from \nhyperparameter sweep', # text to display
             xy=(x_coord, y_coord), # the point to be annotated (arrow tip)
             xytext=(x_coord+0.1, y_coord+0.03), # the position of text (start of arrow)
             arrowprops=dict(facecolor='black', shrink=0.05), # arrow properties
             )
plt.ylabel(r'Macro $F_1$ Score')

plt.savefig('rn18box.png')

