import cv2
import numpy as np

def cam_to_bbox(cam, threshold=0.5):
    cam = (cam > threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(cam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x+w, y+h))

    return boxes