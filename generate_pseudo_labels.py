import sys
sys.path.append("src")

import torch
import os
import cv2

from dataset import PlantDiseaseDataset
from model import get_model
from gradcam import GradCAM
from utils import cam_to_bbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PlantDiseaseDataset("data/raw/plantvillage/PlantVillage")

model = get_model().to(device)
model.load_state_dict(torch.load("outputs/model.pth"))
model.eval()

gradcam = GradCAM(model)

os.makedirs("data/yolo/images", exist_ok=True)
os.makedirs("data/yolo/labels", exist_ok=True)

for img, label, path in dataset:
    img_tensor = img.unsqueeze(0).to(device)

    cam = gradcam.generate(img_tensor)
    boxes = cam_to_bbox(cam)

    img_np = img.permute(1,2,0).numpy() * 255
    img_np = img_np.astype("uint8")

    img_name = os.path.basename(path)

    cv2.imwrite(f"data/yolo/images/{img_name}", img_np)

    h, w = 224, 224

    with open(f"data/yolo/labels/{img_name.replace('.jpg','.txt')}", "w") as f:
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"0 {cx} {cy} {bw} {bh}\n")

print("Pseudo labels generated!")
