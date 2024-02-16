import shutil

for i in range(10):
    folder = f"resnet_advanced/fold_{i}"
    file = f"ResNet18 for fold {i}_loss.png"
    path_from = f"{folder}/{file}"
    path_to = f"media/kfolds/image{i}.png"
    shutil.copy(path_from, path_to)
