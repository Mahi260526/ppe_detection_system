import argparse
import os

try:
    import app  # noqa: F401
except Exception:
    app = None

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the PPE detector on additional edge-case examples."
    )
    parser.add_argument("--data", required=True, help="Path to Ultralytics dataset YAML")
    parser.add_argument("--model", default="models/best.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or 0")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=15, help="Early-stop patience")
    parser.add_argument("--project", default="runs/ppe_finetune", help="Output project directory")
    parser.add_argument("--name", default="bald_head_override", help="Run name")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Base model not found: {args.model}")

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print("Starting fine-tuning run...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        cache=args.cache,
        cos_lr=args.cos_lr,
        lr0=args.lr0,
        pretrained=True,
    )

    run_dir = os.path.join(args.project, args.name)
    best_path = os.path.join(run_dir, "weights", "best.pt")
    print(f"Training complete. Best weights should be at: {best_path}")


if __name__ == "__main__":
    main()
