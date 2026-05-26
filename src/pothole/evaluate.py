#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for YOLOv8 model on the test dataset (potholes).
Supports changing weights, dataset config, split, and imgsz via command line.
"""

import argparse
import sys
import os
import time
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on the dataset (test/val/train).")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="runs/detect/yolo_pothole_yolov8n/weights/best.pt",
        help="Đường dẫn đến file weights (.pt) của mô hình. Mặc định: runs/detect/yolo_pothole_yolov8n/weights/best.pt"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/data.yaml",
        help="Path to dataset config file (data.yaml). Default: data/data.yaml"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        choices=["val", "test", "train"],
        help="Tập dữ liệu để đánh giá (val, test hoặc train). Mặc định: test"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Input image size for evaluation. Default: 640"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16,
        help="Batch size used for evaluation. Default: 16"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="",
        help="Device to use (e.g. '0', 'cpu', 'cuda'). Leave empty to auto-select."
    )
    parser.add_argument(
        "--save-json", 
        action="store_true",
        help="Save evaluation results as COCO JSON."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("      POTHOLE DETECTION MODEL EVALUATION")
    print("=" * 60)
    print(f"[*] Loading model from: {args.weights}")
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        print("Please check the path or choose a different model (e.g. yolo_pothole_yolo26n).")
        sys.exit(1)
        
    if not os.path.exists(args.data):
        print(f"[ERROR] data.yaml not found at: {args.data}")
        sys.exit(1)

    try:
        # Load YOLOv8 model
        model = YOLO(args.weights)
        
        print(f"[*] Starting evaluation on split: {args.split.upper()}")
        print(f"[*] Image size: {args.imgsz}x{args.imgsz} | Batch size: {args.batch}")
        
        start_time = time.time()
        
        # Run validation
        metrics = model.val(
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device if args.device else None,
            save_json=args.save_json,
            plots=True # Draw PR-curve, F1-curve and other plots for visual analysis
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("                 DETAILED EVALUATION RESULTS")
        print("=" * 60)
        
        # Fetch result information
        results_dict = metrics.results_dict
        
        precision = results_dict.get("metrics/precision(B)", 0.0)
        recall = results_dict.get("metrics/recall(B)", 0.0)
        map50 = results_dict.get("metrics/mAP50(B)", 0.0)
        map50_95 = results_dict.get("metrics/mAP50-95(B)", 0.0)
        
        print(f" ❖  Precision                        : {precision * 100:.2f}%")
        print(f" ❖  Recall                           : {recall * 100:.2f}%")
        print(f" ❖  mAP@50 (IoU=0.5)                 : {map50 * 100:.2f}%")
        print(f" ❖  mAP@50-95 (Average mAP)          : {map50_95 * 100:.2f}%")
        print("-" * 60)
        
        # Get speed breakdown
        speed = metrics.speed # contains preprocess, inference, postprocess (ms)
        preprocess_speed = speed.get("preprocess", 0.0)
        inference_speed = speed.get("inference", 0.0)
        postprocess_speed = speed.get("postprocess", 0.0)
        total_latency = preprocess_speed + inference_speed + postprocess_speed
        fps = 1000.0 / total_latency if total_latency > 0 else 0.0
        
        print(" ⚡  PERFORMANCE (Latency per image):")
        print(f"     ➢  Pre-process                   : {preprocess_speed:.2f} ms")
        print(f"     ➢  Inference                     : {inference_speed:.2f} ms")
        print(f"     ➢  Post-process                  : {postprocess_speed:.2f} ms")
        print(f"     ➢  Total latency                 : {total_latency:.2f} ms")
        print(f"     ➢  Estimated throughput (FPS)    : {fps:.2f} FPS")
        print("-" * 60)
        
        # Path to validation results directory
        save_dir = metrics.save_dir
        print(f"[✔] Evaluation completed successfully in {elapsed_time:.2f} seconds!")
        print(f"[✔] Evaluation plots and outputs saved at:")
        print(f"    👉 {save_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
