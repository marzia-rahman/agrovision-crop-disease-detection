# 🌱 AgroVision: Weakly-Supervised Crop Disease Detection using UAV-Inspired Imagery

🚁 *From Leaf-Level Data → Aerial Crop Intelligence*

---

## 📌 Overview

AgroVision is an end-to-end computer vision pipeline for **crop disease detection and localization**, designed to mimic real-world agricultural monitoring systems using UAV (drone) and satellite imagery.

Unlike traditional approaches that rely on expensive bounding box annotations, this project introduces a **weakly-supervised learning framework** that enables disease localization using only image-level labels.

---

## 🚀 Motivation

Early detection of plant diseases is critical for:
- Increasing agricultural yield
- Reducing pesticide overuse
- Enabling precision farming

However, real-world challenges include:

-  Lack of annotated detection datasets  
-  Limited availability of crop-specific data  
-  Differences between lab datasets and field conditions  

This project addresses these challenges by bridging the gap between:

> 🧪 Controlled leaf datasets → 🌾 UAV-scale crop monitoring

---

## 🧠 Key Idea

We convert a **classification problem into a detection problem** using model interpretability.

### Pipeline:
Image
↓
ResNet (Classification)
↓
Grad-CAM (Attention Maps)
↓
Pseudo Bounding Boxes
↓
YOLOv8 (Detection)
↓
Disease Localization




---

### 🎥 Detection Demo

[Dataset](outputs\3.png)
[Detection](outputs\2.png)
[Detection](outputs\1.png)

## ⚙️ Methodology

### 🔹 1. Generalized Classification

- Dataset: PlantVillage (multi-crop)
- Task: Binary classification (Healthy vs Diseased)
- Model: ResNet18 (transfer learning)

👉 Goal: Learn **disease patterns across crops**

---

### 🔥 2. Grad-CAM Interpretability

- Generates attention heatmaps
- Highlights disease-relevant regions

👉 Used for:
- Model explainability  
- Spatial feature extraction  

---

### 🧩 3. Pseudo Label Generation

- Heatmaps are thresholded
- Converted into bounding boxes

👉 Produces:
- Detection labels without manual annotation

---

### 🎯 4. YOLOv8 Detection

- Trained on pseudo labels
- Learns disease localization

👉 Enables real-world deployment capability

---

### 🛰️ 5. UAV Simulation

To approximate aerial imagery:

- Random cropping  
- Multi-scale resizing  
- Gaussian blur  
- Partial leaf visibility  

👉 Improves robustness to real-world conditions

---

## 📂 Project Structure
disease/
├── download_dataset.py
├── filter_dataset.py
├── train_classifier.py
├── generate_pseudo_labels.py
├── train_yolo.py
├── create_demo_gif.py
├── data/
│ ├── raw/
│ ├── yolo/
├── src/
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── gradcam.py
│ ├── utils.py
├── outputs/
│ ├── demo.gif
├── README.md



---

## ⚙️ Installation

### 🔧 Create environment

```bash
conda create -n soybean python=3.10 -y
conda activate soybean

📦 Install dependencies
pip install torch torchvision
pip install ultralytics
pip install opencv-python matplotlib scikit-learn tqdm pillow seaborn imageio kagglehub

📥 Dataset Setup
python download_dataset.py
python filter_dataset.py

1️⃣ Train classifier
python train_classifier.py
2️⃣ Generate pseudo labels
python generate_pseudo_labels.py
3️⃣ Train YOLO detector
python train_yolo.py


---



## 🌍 Real-World Impact

This system can be extended to:

🚁 Drone-based crop monitoring
🛰️ Satellite-based agricultural analytics
🌾 Large-scale disease detection systems
📈 Precision agriculture decision tools


🔮 Future Work
Real UAV dataset integration