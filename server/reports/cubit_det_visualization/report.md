# CUBIT-Det visualization report

| Split | Format | Images | Images with boxes | Boxes | Invalid annotations | Missing images |
|---|---|---:|---:|---:|---:|---:|
| train2017 | YOLO TXT | 3,980 | 3,980 | 12,001 | 2 | 0 |
| val2017 | YOLO TXT | 442 | 442 | 1,326 | 1 | 0 |
| test2017 | YOLO TXT | 1,105 | 1,104 | 3,457 | 0 | 0 |

## Generated figures

- `class_distribution.png`: bounding-box instance counts by class.
- `split_distribution.png`: bounding-box counts by split.
- `<split>_contact_sheet.png`: randomly selected images with boxes.

This is an exploratory visualization, not a complete annotation audit.
Review coordinate conventions and image-label pairing before training.
