import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        self.model.layer4.register_forward_hook(self.forward_hook)
        self.model.layer4.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, img_tensor):
        output = self.model(img_tensor)
        pred = output.argmax()

        self.model.zero_grad()
        output[0, pred].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam