import os
import shutil

SRC = "data/raw/plantvillage/PlantVillage"
DEST = "data/raw/filtered"

os.makedirs(DEST, exist_ok=True)

for folder in os.listdir(SRC):
    src_path = os.path.join(SRC, folder)

    if not os.path.isdir(src_path):
        continue

    dst_path = os.path.join(DEST, folder)
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

print("Filtered dataset ready")
