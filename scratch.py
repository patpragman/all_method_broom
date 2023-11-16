# some code scratch paper to move some things around, not relevant

import os
import shutil

for file in os.listdir("paper_models"):

    if os.path.isdir(f"paper_models/{file}"):
        continue
    elif os.path.basename(file).endswith(".pth"):
        name = os.path.basename(file)
        folder = f"paper_models/{name.split('.')[0]}"
        shutil.move(f"paper_models/{name}", folder)