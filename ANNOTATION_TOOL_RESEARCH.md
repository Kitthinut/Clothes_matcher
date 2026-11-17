# Annotation Tool Research (ANNOTATION_TOOL_RESEARCH.md)

Comparison of labelling approaches for the collaborative annotation workflow
(project spec, section 12). Date: 2026-11-17.

## Requirements

- Bounding boxes (one garment box per item, optional per-attribute crops).
- Custom per-category attribute labels (single-choice and multi-choice).
- JSON export compatible with our `annotations.json` schema v2.
- Multiple annotators working on separate image packages that merge cleanly.
- Zero cost, offline-friendly (project runs fully offline).

## Candidates

| Criterion | Built-in tkinter tool (`1_label/annotate.py`) | Label Studio | CVAT |
|---|---|---|---|
| Install effort | None (part of repo; needs only `pillow`, `opencv-python`) | `pip install label-studio` + browser; server process | Self-host via Docker (heavier) or cvat.ai cloud account |
| Ease of use for friends | Very easy: one script, one window | Medium: account, project setup, task upload | Steepest: Docker/compose or cloud signup |
| Bounding boxes | Yes (drag) | Yes | Yes, best-in-class |
| Custom attributes (single/multi choice) | Yes (native radio/checkboxes per schema JSON) | Yes, via XML labelling config | Yes, via construct settings |
| JSON export | Already our schema | JSON export (needs a converter to our schema) | Export formats (COCO/CSV/JSON) need a converter |
| Multi-annotator packages | Folder-per-friend kits; merge via `merge_annotations.py` | Built-in roles/overlaps | Built-in jobs/stages |
| Offline | Fully | Yes (local server) | Self-host yes; cloud no |
| Cost | Free | Free (community) | Free (CE) / paid cloud |

## Recommendation

**Primary: the built-in tkinter tool.** It natively reads and writes the exact
project schema, needs no installation beyond the existing requirements, works
offline, and the friend-kit flow (`make_kit.py` → label →
`merge_annotations.py`) already satisfies the multi-annotator requirement.

**Secondary (if the built-in tool becomes a bottleneck): Label Studio.**
It is the simplest external option (pip-installable, XML-configurable label
schemes, JSON export). If used, add a converter script from its export format
to `annotations.json` v2 — CVAT is only worth it for very large teams or
video/skeleton annotation, which we do not need.

## Package standard (spec §12.1)

Every annotation package, regardless of tool, must contain:

```
annotation_package/
├── images/            # the images to label
├── annotations.json   # schema v2 (see shared/data.py docstring)
└── README.txt         # package id, annotator id, instructions
```

`make_kit.py` produces exactly this structure; `merge_annotations.py` validates
and merges any package that follows it.
