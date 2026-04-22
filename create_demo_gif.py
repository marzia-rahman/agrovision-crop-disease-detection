import os
import imageio
from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Input images (your pseudo-labeled dataset)
image_dir = "data/yolo/images"

output_frames = "outputs/demo_frames"
os.makedirs(output_frames, exist_ok=True)

frames = []

# Loop through some images
for i, img_name in enumerate(os.listdir(image_dir)[:20]):
    img_path = os.path.join(image_dir, img_name)

    # Run YOLO prediction
    results = model(img_path)

    # Get plotted image (with boxes)
    plotted = results[0].plot()

    # Save frame
    frame_path = os.path.join(output_frames, f"frame_{i}.png")
    cv2.imwrite(frame_path, plotted)

    frames.append(frame_path)

# Create GIF
images = []
for frame in frames:
    images.append(imageio.imread(frame))

gif_path = "outputs/demo.gif"
imageio.mimsave(gif_path, images, duration=0.7)

print(f"✅ GIF saved at {gif_path}")
