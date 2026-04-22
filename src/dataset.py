import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_name))

                # Binary classification
                if "healthy" in class_name.lower():
                    self.labels.append(0)
                else:
                    self.labels.append(1)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(3),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)
        label = self.labels[idx]

        return img, label, self.images[idx]