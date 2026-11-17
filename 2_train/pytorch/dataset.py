"""Image pipeline - PyTorch (2_train/pytorch/dataset.py).

Input per sample: the garment-box CROP (schema v2 stores the box in original
pixels), resized to the shared 288x384 input and ImageNet-normalised —
identical to the inference preprocessing (shared/preprocessing.py).

Augmentation (train only): random horizontal flip + slight colour jitter.
Theory: augmentation = free extra training samples, reduces overfitting.

Targets (shared/data.encode_labels):
  single attribute -> long tensor (class index, -1 = missing)
  multi attribute  -> float tensor (multi-hot; all -1.0 = missing,
                     all 0.0 = explicit "none of these")
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]  # 2_train/pytorch/ -> project root
PY = Path(__file__).resolve().parent
for p in (ROOT, PY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.data import encode_labels  # noqa: E402
from shared.preprocessing import INPUT_H, INPUT_W, MEAN, STD, garment_crop  # noqa: E402

# ImageNet normalisation keeps the pretrained weights happy
NORM = T.Normalize(MEAN, STD)
TRAIN_TF = T.Compose([
    T.Resize((INPUT_H, INPUT_W)),  # torchvision takes (height, width)
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    T.ToTensor(),
    NORM,
])
EVAL_TF = T.Compose([T.Resize((INPUT_H, INPUT_W)), T.ToTensor(), NORM])


def target_tensors(attributes, spec):
    """Attributes dict -> {key: tensor} using encode_labels conventions."""
    encoded = encode_labels(attributes, spec)
    out = {}
    for k, cat in spec.items():
        if cat["mode"] == "multi":
            v = encoded[k]
            if isinstance(v, list):
                hot = torch.zeros(len(cat["options"]), dtype=torch.float32)
                for i in v:
                    hot[i] = 1.0
            else:  # -1 -> missing: an all -1 sentinel the loss masks out
                hot = torch.full((len(cat["options"]),), -1.0, dtype=torch.float32)
            out[k] = hot
        else:
            out[k] = torch.tensor(encoded[k], dtype=torch.long)
    return out


class GarmentDataset(Dataset):
    def __init__(self, samples, spec, train=True):
        self.samples, self.spec = samples, spec
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["path"]).convert("RGB")
        x = self.tf(garment_crop(img, s.get("box")))
        return x, target_tensors(s["attributes"], self.spec)
