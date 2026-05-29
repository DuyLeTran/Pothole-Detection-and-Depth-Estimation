import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/fine-tune a YOLOv8 model")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights/checkpoint")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="yolo_pothole_yolov8n")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--cos_lr", type=bool, default=True, help="Cosine learning rate")
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / path).resolve())

def main() -> None:
    args = parse_args()
    # Use YOLOv8-nano (yolov8n.pt) to target >= 20 FPS on Edge CPU
    model = YOLO(resolve_repo_path(args.weights))
    
    # Configuration for a large, diverse dataset (many images, various conditions: night, rain, harsh sunlight)
    model.train(
        data=resolve_repo_path(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,          # 16 is a safe GPU batch size, large enough for stable gradients on big datasets
        name=args.name,
        freeze=args.freeze,
        # --- TECHNIQUES FOR LARGE DATASETS ---
        # Do not freeze layers so the model can fully leverage the large dataset
        
        # Optimizer: with sufficient data, the default optimizer (often SGD) can generalize better than AdamW
        optimizer="auto",    
        lr0=args.lr0,            
        lrf=args.lrf,            
        weight_decay=args.weight_decay, # keep a small weight decay; large dataset helps prevent overfitting
        cos_lr=args.cos_lr,         
        
        # --- STRONGER DATA AUGMENTATION ---
        # 1. Simulate lighting conditions (night, rain, harsh sunlight)
        hsv_h=0.015,         
        hsv_s=0.7,           # saturation perturbation (rain/night often desaturated)
        hsv_v=0.4,           # value/brightness perturbation (important for night/bright simulation)

        # 2. Geometric & spatial augmentations
        scale=0.5,           # simulate near/far scales
        translate=0.1,       
        degrees=10.0,        # allow up to 10 degrees rotation to simulate vibration or camera tilt

        # 3. Occlusion & mix techniques
        erasing=0.2,         # random erasing to handle partial occlusions (puddles, shadows)
        mosaic=1.0,          # mosaic augmentation to increase context
        mixup=0.15,          # 15% mixup for harder overlapping/noisy examples
        fliplr=0.5,          

        warmup_epochs=3.0,
    )

if __name__ == "__main__":
    main()