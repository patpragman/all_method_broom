import os
from statistics import stdev, mean
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend had to do this for support on different computers initially, YMMV

import matplotlib.pyplot as plt
import pandas as pd

f1s = []
for fold in os.listdir("resnet_advanced"):
    if fold == "final":
        continue

    with open(f"resnet_advanced/{fold}/report.md", "r") as fold_data:
        lines = fold_data.readlines()
        print(lines[7])
        macro_avg = float(lines[7].split()[-2])
        f1s.append(macro_avg)


print('average f1 score:')
print(mean(f1s))
print("standard deviation:")
print(stdev(f1s))

px = 1
py = 0.78

arx = 1.1
ary = 0.79

ardx = px - arx
ardy = py - ary

df = pd.DataFrame({"macro f1":f1s})
df.boxplot()
plt.arrow(arx, ary, ardx, ardy, length_includes_head=True, width=0.0001)
plt.plot(px, py, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")

plt.text(arx, ary, r"$F_1$ from paper")
plt.ylabel(r'Macro $F_1$ Score')
plt.title('Box Plot of K-Folds Cross Validation on ResNet-18')
plt.savefig('boxy.png')