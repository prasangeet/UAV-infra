# CUBIT-Det tiled dataset visualization report

| Split | Tiles | Tiles with boxes | Boxes | Invalid label rows | Missing label files |
|---|---:|---:|---:|---:|---:|
| train | 23,183 | 15,223 | 18,174 | 0 | 0 |
| val | 2,561 | 1,677 | 1,942 | 0 | 0 |
| test | 6,653 | 4,443 | 5,356 | 0 | 0 |

## Class distribution

| Class | Instances |
|---|---:|
| Crack | 20,992 |
| Spalling | 2,827 |
| Moisture | 1,653 |

## Generated figures

- `class_distribution.png`: object instances by class.
- `split_distribution.png`: object instances by split.
- `<split>_contact_sheet.png`: sampled tiles with bounding boxes.

This is an exploratory visualization, not a complete annotation audit.
Inspect clipped boxes and tile boundaries before training.
