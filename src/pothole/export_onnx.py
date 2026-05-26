#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export YOLOv8 PyTorch model (.pt) to ONNX format (.onnx).
Supports custom image size, dynamic input shapes, and graph simplification.
"""

import argparse
import sys
import os
import time
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch model (.pt) to ONNX (.onnx).")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="runs/detect/yolo_pothole_yolov8n/weights/best.pt",
        help="Path to PyTorch model weights (.pt). Default: runs/detect/yolo_pothole_yolov8n/weights/best.pt"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Image size for the exported ONNX model. Default: 640"
    )
    parser.add_argument(
        "--dynamic", 
        action="store_true",
        help="Enable dynamic input shapes (width and height can vary during inference)."
    )
    parser.add_argument(
        "--simplify", 
        action="store_false", # True by default, add --simplify to set False (or vice versa, we use simplify=True as default)
        dest="simplify",
        help="Disable ONNX graph simplification (onnx-simplifier is required)."
    )
    parser.add_argument(
        "--half", 
        action="store_true",
        help="Export with FP16 half precision (recommended for GPU deployment)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("      YOLOv8 MODEL EXPORT TO ONNX")
    print("=" * 60)
    print(f"[*] Loading PyTorch weights from: {args.weights}")
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        sys.exit(1)
        
    try:
        # Load the PyTorch YOLO model
        model = YOLO(args.weights)
        
        print(f"[*] Export configuration:")
        print(f"    ➢ Target format: ONNX")
        print(f"    ➢ Image size   : {args.imgsz}x{args.imgsz}")
        print(f"    ➢ Dynamic shape: {args.dynamic}")
        print(f"    ➢ Simplify graph: {args.simplify}")
        print(f"    ➢ FP16 Half     : {args.half}")
        print("-" * 60)
        print("[*] Starting export process... (This might take a moment)")
        
        start_time = time.time()
        
        # Export the model
        exported_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            simplify=args.simplify,
            half=args.half
        )
        
        elapsed_time = time.time() - start_time
        
        print("-" * 60)
        print(f"[✔] Model exported successfully in {elapsed_time:.2f} seconds!")
        print(f"[✔] Exported ONNX model saved at:")
        print(f"    👉 {exported_path}")
        print("=" * 60)
        print("\n💡 You can now run inference with this ONNX model using:")
        print(f"   python src/pothole/inference.py --weights \"{exported_path}\" --source \"<your_input>\" --save")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] An error occurred during model export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
