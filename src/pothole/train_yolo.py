from ultralytics import YOLO

def main() -> None:
    # Use YOLOv8-nano (yolov8n.pt) to target >= 20 FPS on Edge CPU
    model = YOLO("yolov8n.pt")
    
    # Configuration for a large, diverse dataset (many images, various conditions: night, rain, harsh sunlight)
    model.train(
        data="data/data.yaml",
        epochs=400,        
        patience=50,       
        imgsz=640, 
        batch=16,          # 16 is a safe GPU batch size, large enough for stable gradients on big datasets
        name="yolo_pothole_yolov8n",
        
        # --- TECHNIQUES FOR LARGE DATASETS ---
        # Do not freeze layers so the model can fully leverage the large dataset
        
        # Optimizer: with sufficient data, the default optimizer (often SGD) can generalize better than AdamW
        optimizer="auto",    
        lr0=0.01,            
        lrf=0.01,            
        weight_decay=0.0005, # keep a small weight decay; large dataset helps prevent overfitting
        cos_lr=True,         
        
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
    )

if __name__ == "__main__":
    main()