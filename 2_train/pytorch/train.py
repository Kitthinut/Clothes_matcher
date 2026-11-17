"""Training loop - PyTorch (2_train/pytorch/train.py).

Theory:
- AMP mixed precision: float16 matmuls on GPU = faster, less VRAM.
- Cosine LR schedule: learning rate decays smoothly to ~0 (SGDR paper).
- AdamW + backbone/head LR split: pretrained backbone gets 10x smaller LR
  so its learned features are not destroyed early in training.
- Early stopping: keep the best validation score, stop after N flat epochs.

Artifacts (spec section 5.3): weights/<garment>_pytorch.pt holds the state
dict + spec + config (arch, input size, preprocessing, versions, args);
training history goes to <...>.csv; final test metrics (accuracy/precision/
recall/F1 per attribute) to <...>_metrics.json.

Usage:  python 2_train/pytorch/train.py --garment top
"""

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # 2_train/pytorch/ -> project root
PY = Path(__file__).resolve().parent
for p in (ROOT, PY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import GarmentDataset  # noqa: E402
from model import MultiHeadNet, loss_fn  # noqa: E402
from evaluation.metrics import attribute_metrics, format_report, mean_accuracy  # noqa: E402
from shared.data import load_split  # noqa: E402
from shared.preprocessing import config  # noqa: E402
from shared.schema import load_spec  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


def to_labels(tensors, spec):
    """Batch target tensors -> per-sample attribute dicts (metrics format)."""
    out = []
    for i in range(len(next(iter(tensors.values())))):
        sample = {}
        for k, cat in spec.items():
            t = tensors[k][i]
            if cat["mode"] == "multi":
                sample[k] = [] if (t < 0).all() else [
                    cat["options"][j] for j in (t > 0.5).nonzero().flatten().tolist()]
            else:
                v = int(t)
                sample[k] = cat["options"][v] if 0 <= v < len(cat["options"]) else None
        out.append(sample)
    return out


def pred_labels(logits, spec):
    """Logits -> per-sample predicted attribute dicts
    (single: argmax; multi: sigmoid > 0.5, falling back to the top option)."""
    out = []
    for i in range(len(next(iter(logits.values())))):
        sample = {}
        for k, cat in spec.items():
            if cat["mode"] == "multi":
                picked = [cat["options"][j] for j in
                          (torch.sigmoid(logits[k][i]) > 0.5).nonzero().flatten().tolist()]
                sample[k] = picked or [cat["options"][int(logits[k][i].argmax())]]
            else:
                sample[k] = cat["options"][int(logits[k][i].argmax())]
        out.append(sample)
    return out


@torch.no_grad()
def evaluate(model, loader, spec, dev):
    """Loss + per-attribute metrics (accuracy/precision/recall/F1)."""
    model.eval()
    total, n = 0.0, 0
    truth, preds = [], []
    for x, y in loader:
        x, y = x.to(dev), {k: v.to(dev) for k, v in y.items()}
        out = model(x)
        total += loss_fn(out, y, spec).item() * len(x)
        n += len(x)
        truth += to_labels(y, spec)
        preds += pred_labels(out, spec)
    metrics = attribute_metrics(truth, preds, spec)
    return total / max(n, 1), mean_accuracy(metrics), metrics


def save_metrics_json(path, metrics, args, extra=None):
    import torch  # versions for the artifact (spec section 5.3)
    import torchvision
    data = {
        "garment": args.garment, "backend": "pytorch",
        "metrics": metrics, "mean_accuracy": mean_accuracy(metrics),
        "versions": {"torch": torch.__version__,
                     "torchvision": torchvision.__version__,
                     "python": sys.version.split()[0],
                     "platform": platform.platform()},
        "args": {k: v for k, v in vars(args).items()},
    }
    data.update(extra or {})
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Metrics -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment", choices=("top", "bottom"), default="top")
    ap.add_argument("--test-only", action="store_true", help="score the test split, no training")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)  # 0 is safest on Windows
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--show", type=int, default=0,
                    help="popup with Grad-CAM boxes on N random images when done")
    ap.add_argument("--save-visual", default=None,
                    help="with --show: save PNGs to this folder instead of popup")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.test_only:
        test_only(a)
        return

    set_seed(a.seed)
    spec = load_spec(a.garment)
    splits = ROOT / "dataset" / a.garment / "splits"
    data = {n: load_split(splits / f"{n}.json") for n in ("train", "valid", "test")}
    if not data["train"]:
        raise SystemExit(f"No data. Run: python shared/split.py --garment {a.garment}")

    dev = get_device()
    tl = DataLoader(GarmentDataset(data["train"], spec, True), batch_size=a.batch,
                    shuffle=True, num_workers=a.workers)
    vl = DataLoader(GarmentDataset(data["valid"], spec, False), batch_size=a.batch,
                    num_workers=a.workers)

    model = MultiHeadNet(spec).to(dev)
    backbone = [p for n_, p in model.named_parameters() if "backbone" in n_]
    heads = [p for n_, p in model.named_parameters() if "backbone" not in n_]
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": a.lr * 0.1}, {"params": heads, "lr": a.lr}],
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.epochs - 1))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    out = ROOT / "weights" / f"{a.garment}_pytorch.pt"
    out.parent.mkdir(exist_ok=True)
    log = out.with_suffix(".csv")
    cfg = config(a.garment, "resnet18", extra={
        "args": {k: v for k, v in vars(a).items()},
        "num_samples": {n: len(v) for n, v in data.items()},
        "versions": {"torch": torch.__version__,
                     "torchvision": torchvision.__version__,
                     "python": sys.version.split()[0]},
    })
    best, bad = -1.0, 0
    with open(log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for x, y in tl:
            x, y = x.to(dev), {k: v.to(dev) for k, v in y.items()}
            with torch.amp.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(x), y, spec)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(x)
        sched.step()

        vloss, vacc, metrics = evaluate(model, vl, spec, dev)
        print(f"Epoch {ep:02d} | train {run / len(data['train']):.4f} | "
              f"val {vloss:.4f} | acc {'N/A' if vacc is None else f'{vacc:.3f}'}")
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{run / len(data['train']):.4f}",
                                    f"{vloss:.4f}",
                                    f"{vacc:.4f}" if vacc is not None else ""])

        if vacc is not None and vacc > best:
            best, bad = vacc, 0
            torch.save({"state": model.state_dict(), "spec": spec, "config": cfg}, out)
        elif vacc is None:
            # no validation data (e.g. tiny set): keep the latest epoch instead
            torch.save({"state": model.state_dict(), "spec": spec, "config": cfg}, out)
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                print(f"Early stopping (no improvement for {a.patience} epochs)")
                break

    print(f"Best val acc {best:.3f} -> {out}")

    # final test-split check + metrics artifact
    test_only(a)
    if a.show:
        sys.path.insert(0, str(PY))  # local visualize.py (shared/visualize.py also exists)
        from visualize import popup_from_samples
        shown = random.sample(data["valid"], min(a.show, len(data["valid"])))
        popup_from_samples(shown, a.garment, a.save_visual)


@torch.no_grad()
def test_only(args):
    """Final unbiased check on the never-touched test split.
    Prints a per-attribute report and writes <garment>_pytorch_metrics.json."""
    spec = load_spec(args.garment)
    splits = ROOT / "dataset" / args.garment / "splits"
    test = load_split(splits / "test.json")
    if not test:
        raise SystemExit("No test split. Run: python shared/split.py --garment <g>")
    ck = torch.load(ROOT / "weights" / f"{args.garment}_pytorch.pt", map_location="cpu",
                    weights_only=False)
    model = MultiHeadNet(ck["spec"], pretrained=False)
    model.load_state_dict(ck["state"])
    dev = get_device()
    model.to(dev)
    vl = DataLoader(GarmentDataset(test, spec, False), batch_size=args.batch)

    t0 = time.perf_counter()
    loss, acc, metrics = evaluate(model, vl, spec, dev)
    elapsed = (time.perf_counter() - t0) / max(len(test), 1)

    print(f"TEST loss {loss:.4f} | mean acc "
          f"{'N/A' if acc is None else f'{acc:.3f}'} | "
          f"{elapsed * 1000:.0f} ms/image")
    print(format_report(metrics))
    save_metrics_json(ROOT / "weights" / f"{args.garment}_pytorch_metrics.json",
                      metrics, args,
                      extra={"test_loss": loss, "seconds_per_image": round(elapsed, 4)})
    if args.show:
        sys.path.insert(0, str(PY))  # local visualize.py (shared/visualize.py also exists)
        from visualize import popup_from_samples
        shown = random.sample(test, min(args.show, len(test)))
        popup_from_samples(shown, args.garment, args.save_visual)


if __name__ == "__main__":
    main()
