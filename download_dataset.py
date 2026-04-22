import kagglehub
import shutil
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("emmarex/plantdisease")

DEST = "data/raw/plantvillage"
os.makedirs(DEST, exist_ok=True)

for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(DEST, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset ready at:", DEST)
