"""Shared evaluation metrics (evaluation/metrics.py).

Pure python — no backend dependencies, so PyTorch and TensorFlow
experiments are scored by the exact same code (reproducible comparison,
project spec section 5.2).

Per attribute category (key):
  accuracy  — exact prediction match rate
              (single: same class; multi: identical option set)
  precision / recall / f1 — macro-averaged over the schema options
              (option-level one-vs-rest, then averaged)
  support   — number of labelled samples for that key
"""

from collections import defaultdict


def _as_set(v):
    return set(v) if isinstance(v, (list, tuple, set)) else {v}


def attribute_metrics(truth, preds, spec):
    """truth, preds: parallel lists of {key: value | [values]} dicts.
    Returns {key: {accuracy, precision, recall, f1, support}}.
    Samples missing a key (None / -1) are skipped for that key only."""
    out = {}
    for key, cat in spec.items():
        pairs = [(t, p) for t, p in zip(truth, preds)
                 if t.get(key) is not None and t.get(key) != -1
                 and p.get(key) is not None]
        support = len(pairs)
        if not support:
            out[key] = {"accuracy": None, "precision": None, "recall": None,
                        "f1": None, "support": 0}
            continue

        correct = sum(_as_set(t[key]) == _as_set(p[key]) for t, p in pairs)

        # option-level one-vs-rest counts (TP / FP / FN), then macro-average
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        for t, p in pairs:
            t_set, p_set = _as_set(t[key]), _as_set(p[key])
            for opt in t_set | p_set:
                if opt in t_set and opt in p_set:
                    tp[opt] += 1
                elif opt in p_set:
                    fp[opt] += 1
                else:
                    fn[opt] += 1

        def macro(denom_key):
            vals = []
            for opt in cat["options"]:
                d = tp[opt] + (fp[opt] if denom_key == "fp" else fn[opt])
                vals.append(tp[opt] / d if d else 0.0)
            return sum(vals) / len(vals) if vals else 0.0

        precision, recall = macro("fp"), macro("fn")
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        out[key] = {"accuracy": correct / support, "precision": precision,
                    "recall": recall, "f1": f1, "support": support}
    return out


def mean_accuracy(metrics):
    """{key: metrics} -> mean accuracy over keys that have support (or None)."""
    vals = [m["accuracy"] for m in metrics.values() if m["accuracy"] is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(metrics):
    """One-line console summary: 'neck 0.92 | sleeves 0.85 | ...'."""
    parts = []
    for k, m in metrics.items():
        acc = m["accuracy"]
        parts.append(f"{k}={'N/A' if acc is None else format(acc, '.2f')}")
    return " | ".join(parts)


def format_report(metrics):
    """Multi-line printable per-attribute report."""
    lines = [f"{'attribute':<14} {'support':>7} {'accuracy':>8} "
             f"{'precision':>9} {'recall':>7} {'f1':>6}"]
    for k, m in metrics.items():
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"{k:<14} {m['support']:>7} {fmt(m['accuracy']):>8} "
                     f"{fmt(m['precision']):>9} {fmt(m['recall']):>7} {fmt(m['f1']):>6}")
    return "\n".join(lines)
