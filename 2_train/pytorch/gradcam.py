"""Grad-CAM - PyTorch (2_train/pytorch/gradcam.py).

Theory (Grad-CAM, Selvaraju et al. 2017):
For each attribute head, take the gradients of the predicted class score with
respect to the last conv feature map. Average them per channel to get channel
importance weights; the ReLU'd weighted sum of the feature channels is a
heatmap of WHERE the model looked for that attribute. shared/visualize.py
turns the heatmap into a box, so no box labels are needed for training.
"""

import cv2
import torch
import torch.nn.functional as F

from shared.visualize import bbox_from_heat


class GradCAM:
    """Hooks the last conv layer of MultiHeadNet once, then one backward pass
    per head produces the heatmap for that head."""

    def __init__(self, model):
        self.model = model
        self.acts = self.grads = None
        last = model.backbone_conv[-1]  # ResNet18 layer4, output (B,512,7,7)
        last.register_forward_hook(self._save_act)
        last.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.acts = output.detach()

    def _save_grad(self, _, __, grad_output):
        self.grads = grad_output[0].detach()

    def boxes(self, x):
        """(1,C,H,W) input -> {head key: (x1,y1,x2,y2) px box or None} in
        INPUT-PIXEL coordinates of x. Call right after a forward pass;
        one backward per head."""
        self.model.eval()
        h, w = int(x.shape[-2]), int(x.shape[-1])
        with torch.enable_grad():
            logits = self.model(x)
            out = {}
            for k in logits:  # one backward per attribute head
                self.model.zero_grad(set_to_none=True)
                logits[k][0].max().backward(retain_graph=True)
                # channel importance = spatial mean of the gradients (GAP)
                wch = self.grads[0].mean(dim=(1, 2))  # (512,)
                cam = F.relu((wch[:, None, None] * self.acts[0]).sum(0))
                cam = cam.detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                out[k] = bbox_from_heat(cv2.resize(cam, (w, h)))
        return out
