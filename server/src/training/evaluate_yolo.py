
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt"
)

DATA_YAML = PROJECT_ROOT / "data/cubit_det/data.yaml"

IMAGE_SIZE = 1024
BATCH_SIZE = 4
DEVICE = 0
WORKERS = 4

PROJECT_DIR = PROJECT_ROOT / "runs/detect"
RUN_NAME = "cubit_det_yolo11n_1024_finetune50_test"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    print("=" * 65)
    print("YOLO11n — Fine-tuned Model Test Evaluation")
    print("=" * 65)
    print(f"Model       : {MODEL_PATH}")
    print(f"Dataset     : {DATA_YAML}")
    print("Split       : test")
    print(f"Image size  : {IMAGE_SIZE}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Device      : {DEVICE}")
    print("=" * 65)

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        plots=True,
        save_json=True,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(f"Precision (Box): {metrics.box.mp:.4f}")
    print(f"Recall (Box)   : {metrics.box.mr:.4f}")
    print(f"mAP@50         : {metrics.box.map50:.4f}")
    print(f"mAP@50-95      : {metrics.box.map:.4f}")

    print("\nPER-CLASS RESULTS")

    for class_id, class_name in model.names.items():
        precision = metrics.box.p[class_id]
        recall = metrics.box.r[class_id]
        map50 = metrics.box.ap50[class_id]
        map5095 = metrics.box.ap[class_id]

        print(
            f"{class_name:<15}"
            f"P: {precision:.4f}  "
            f"R: {recall:.4f}  "
            f"mAP50: {map50:.4f}  "
            f"mAP50-95: {map5095:.4f}"
        )

    print("\nSpeed (ms/image)")
    for key, value in metrics.speed.items():
        print(f"{key:<15}: {value:.3f}")

    print(f"\nEvaluation artifacts: {metrics.save_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
