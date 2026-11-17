"""Multi-head garment classifier - PyTorch (2_train/pytorch/model.py).

Theory:
- Transfer learning: a ResNet18 pretrained on ImageNet provides the visual
  features; we only train the small classifier heads on top (fine-tuning).
- Multi-task learning: one shared backbone, one Linear head per attribute.
  Single-label heads use Softmax + Cross-Entropy (ignore_index for missing);
  multi-label heads (e.g. pattern) use sigmoid + BCE so an item can be
  "dotted AND striped". Missing multi labels (sentinel -1 rows) are masked.
"""

import torch
import torch.nn as nn
from torchvision import models


class MultiHeadNet(nn.Module):
    def __init__(self, spec, pretrained=True):
        super().__init__()
        self.spec = spec
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        # children[:-2] keeps everything up to layer4 -> output is a spatial
        # feature map (B,512,7,7); needed later by Grad-CAM (gradcam.py)
        self.backbone_conv = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        dim = backbone.fc.in_features  # 512 for ResNet18
        # "h_" prefix: keys like "type" collide with nn.Module.type(), the
        # prefix keeps ModuleDict happy (forward/loss still use plain keys)
        self.heads = nn.ModuleDict(
            {f"h_{k}": nn.Linear(dim, len(c["options"])) for k, c in spec.items()})

    def forward(self, x):
        fmap = self.backbone_conv(x)      # (B,512,7,7) spatial features
        z = self.flat(self.pool(fmap))    # (B,512) for the heads
        # strip the "h_" prefix so logits use the plain schema keys
        return {k[2:]: head(z) for k, head in self.heads.items()}

    def embed(self, x):
        """Penultimate features (B,512) — the visual embedding used by the
        closet similarity search. Same tensor the heads consume, so it costs
        nothing extra and lives in the same space for all closet items
        processed with this checkpoint."""
        fmap = self.backbone_conv(x)
        return self.flat(self.pool(fmap))


def loss_fn(logits, labels, spec):
    """Mean loss over heads.
    single: CrossEntropy, ignore_index=-1 (missing attribute).
    multi:  BCEWithLogits on the multi-hot target; rows whose sentinel is
            -1 (missing) are masked out so they contribute nothing."""
    losses = []
    for k, cat in spec.items():
        if cat["mode"] == "multi":
            target = labels[k].float()
            mask = (target >= 0).all(dim=1)
            if not mask.any():
                continue
            per_elem = nn.functional.binary_cross_entropy_with_logits(
                logits[k], target.clamp(min=0.0), reduction="none")
            per_sample = per_elem.mean(dim=1)          # (B,)
            losses.append(per_sample[mask].mean())
        else:
            if (labels[k] >= 0).any():
                losses.append(nn.functional.cross_entropy(
                    logits[k], labels[k], ignore_index=-1))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def decode(logits, spec):
    """Logits dict {key: (1, n)} -> {key: {option: prob}}.
    single: softmax; multi: sigmoid per option."""
    out = {}
    for k, cat in spec.items():
        lg = logits[k][0].detach().cpu()
        if cat["mode"] == "multi":
            probs = torch.sigmoid(lg)
        else:
            probs = torch.softmax(lg, dim=0)
        out[k] = {opt: p.item() for opt, p in zip(cat["options"], probs)}
    return out


def labels_from_probs(probs, spec, thresh=0.5):
    """Probs dict -> predicted labels {key: str | list[str]}.
    multi: every option with prob >= thresh (never empty: falls back to the
    single most likely option so downstream code always has a label)."""
    labels = {}
    for k, cat in spec.items():
        p = probs[k]
        if cat["mode"] == "multi":
            picked = sorted(o for o, v in p.items() if v >= thresh)
            labels[k] = picked or [max(p, key=p.get)]
        else:
            labels[k] = max(p, key=p.get)
    return labels
