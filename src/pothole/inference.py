#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for detecting potholes on Image, Video or Webcam.
Includes a semi-transparent HUD (Heads-Up Display) that shows:
- A dynamic Safety/Alert status (red/green).
- The number of detected potholes in real-time.
- Inference FPS and the hardware device in use.
"""

import argparse
import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference to detect potholes on Image/Video/Webcam.")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="runs/detect/yolo_pothole_yolov8n/weights/best.pt",
        help="Path to weights file (.pt). Default: runs/detect/yolo_pothole_yolov8n/weights/best.pt"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        required=True,
        help="Input source: image file, video file, image directory, or '0'/'1' for Webcam."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25,
        help="Confidence threshold. Default: 0.25"
    )
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (NMS). Default: 0.45"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Inference image size. Default: 640"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="",
        help="Thiết bị chạy (ví dụ: '0', 'cpu', 'cuda'). Để trống để tự động chọn."
    )
    parser.add_argument(
        "--save", 
        action="store_true", # default is not to save, add --save to enable
        help="Save output (images/videos with boxes and HUD)."
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="runs/inference",
        help="Directory to save results. Default: runs/inference"
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show OpenCV preview window (not recommended on headless VPS/SSH)."
    )
    parser.add_argument(
        "--hud-off", 
        action="store_true",
        help="Disable HUD overlay; draw only basic YOLO bounding boxes."
    )
    return parser.parse_args()

def draw_hud(frame, pothole_count, fps, device_name):
    """
    Draw a semi-transparent HUD banner and high-level alerts on the frame.
    """
    h, w = frame.shape[:2]
    
    # 1. Create a semi-transparent horizontal HUD banner at the top (70px height)
    hud_height = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 15, 15), -1) # dark gray overlay
    
    # Apply transparency alpha = 0.75
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 2. Draw a subtle bottom separator line for the HUD
    cv2.line(frame, (0, hud_height), (w, hud_height), (225, 225, 30), 1)

    # 3. Draw system logo/name (left)
    cv2.putText(frame, "MET EV | POTHOLE RADAR", (20, 28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw hardware name under the logo
    dev_text = f"Hardware: {device_name.upper()}"
    cv2.putText(frame, dev_text, (20, 52), 
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # 4. Draw status & pothole counter (center)
    if pothole_count > 0:

        status_text = f" DANGER: DETECTED {pothole_count} POTHOLE{'S' if pothole_count > 1 else ''} "
        status_color = (0, 0, 230) # Bright red for danger
        text_color = (255, 255, 255)
    else:
        # Safe status in green
        status_text = " STATUS: SAFE ROAD "
        status_color = (0, 180, 0) # Bright green for safe
        text_color = (255, 255, 255)
        
    # Compute background size for the status label to center it on the HUD
    (tw, th), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    cx = w // 2
    x1, y1 = cx - (tw // 2) - 10, 15
    x2, y2 = cx + (tw // 2) + 10, 48
    
    # Draw background rectangle for the status and render the text
    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, -1)
    cv2.putText(frame, status_text, (x1 + 10, y1 + 22), 
                cv2.FONT_HERSHEY_DUPLEX, 0.55, text_color, 1, cv2.LINE_AA)

    # 5. Draw processing FPS (right)
    fps_text = f"FPS: {fps:.1f}"
    # Change FPS color: >20 FPS (green - meets Edge CPU target), <20 FPS (orange)
    fps_color = (0, 255, 0) if fps >= 20.0 else (0, 180, 255)
    cv2.putText(frame, fps_text, (w - 120, 28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, fps_color, 1, cv2.LINE_AA)
    
    # Show estimated latency
    latency = 1000.0 / fps if fps > 0 else 0.0
    lat_text = f"Latency: {latency:.1f} ms"
    cv2.putText(frame, lat_text, (w - 120, 52), 
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return frame

def draw_custom_boxes(image, boxes):
    """
    Draw bounding boxes and only display confidence instead of class name.
    """
    for box in boxes:
        # Get coordinates (xyxy)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Get confidence
        conf = float(box.conf[0])
        
        # Bounding box bright red to highlight potholes
        color = (0, 0, 255) 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Only show confidence (e.g., "0.85")
        label_text = f"{conf:.2f}"
        
        # Draw background for the confidence label
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        tx1, ty1 = x1, max(y1 - th - 6, 0)
        tx2, ty2 = x1 + tw + 8, max(y1, th + 6)
        cv2.rectangle(image, (tx1, ty1), (tx2, ty2), color, -1)
        
        # Draw the confidence text in white
        cv2.putText(image, label_text, (tx1 + 4, ty1 + th + 2), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    return image

def process_image(model, source_path, args, device_name):
    """
    Process inference on a single image file.
    """
    print(f"[*] Processing image: {source_path}")
    
    # Run YOLOv8 inference
    results = model.predict(
        source=source_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        verbose=False
    )
    
    result = results[0]
    pothole_count = len(result.boxes)
    
    # Make a copy of the original image to draw custom boxes (no class names, only confidence)
    annotated_img = result.orig_img.copy()
    annotated_img = draw_custom_boxes(annotated_img, result.boxes)
    
    # Add HUD to the image (for images, FPS is N/A or computed from inference time)
    if not args.hud_off:
        speed = result.speed
        total_time = speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0)
        fps = 1000.0 / total_time if total_time > 0 else 0.0
        annotated_img = draw_hud(annotated_img, pothole_count, fps, device_name)
        
    # Show if requested
    if args.show:
        cv2.imshow("MET EV Pothole Detection - Image Preview", annotated_img)
        print("[*] Press any key in the preview window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # Save results
    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
        out_name = f"predict_{Path(source_path).name}"
        out_path = os.path.join(args.save_dir, out_name)
        cv2.imwrite(out_path, annotated_img)
        print(f"[✔] Saved result image at: {out_path}")

def process_video(model, source_path, args, device_name):
    """
    Process inference on a video file or webcam/RTSP stream.
    """
    is_webcam = source_path.isdigit()
    
    if is_webcam:
        cap_source = int(source_path)
        print(f"[*] Initializing Webcam ID: {cap_source}...")
    else:
        cap_source = source_path
        print(f"[*] Opening video file: {source_path}...")
        
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video/webcam source: {source_path}")
        return

    # Get source video parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0 # fallback if FPS cannot be retrieved
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    
    # Initialize VideoWriter if saving results
    out_writer = None
    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
        if is_webcam:
            out_name = f"webcam_predict_{int(time.time())}.mp4"
        else:
            out_name = f"predict_{Path(source_path).name}"
            # Ensure output uses .mp4 extension to avoid codec issues
            if not out_name.endswith(".mp4"):
                out_name = f"{Path(out_name).stem}.mp4"
                
        out_path = os.path.join(args.save_dir, out_name)
        
        # Use common MP4V codec across platforms
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        print(f"[*] Bắt đầu ghi video kết quả lưu tại: {out_path}")

    print("[*] Running inference... Press 'q' in preview window (if shown) to stop early.")
    
    frame_idx = 0
    prev_time = time.time()
    avg_fps = 0.0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Run YOLOv8 on the frame
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device if args.device else None,
                verbose=False
            )
            
            result = results[0]
            pothole_count = len(result.boxes)
            
            # Draw custom bounding boxes (confidence only) on the original frame
            annotated_frame = result.orig_img.copy()
            annotated_frame = draw_custom_boxes(annotated_frame, result.boxes)
            
            # Compute actual processing FPS
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            
            # Smooth FPS for less jittery display
            instant_fps = 1.0 / exec_time if exec_time > 0 else 30.0
            if avg_fps == 0.0:
                avg_fps = instant_fps
            else:
                avg_fps = 0.9 * avg_fps + 0.1 * instant_fps
            
            # Draw HUD onto the frame
            if not args.hud_off:
                annotated_frame = draw_hud(annotated_frame, pothole_count, avg_fps, device_name)
                
            # Write annotated frame to output video
            if out_writer is not None:
                out_writer.write(annotated_frame)
                
            # Show preview window
            if args.show:
                cv2.imshow("MET EV Pothole Detection - Video Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] Stop command received. Exiting loop...")
                    break
                    
            # Print progress for file video
            if not is_webcam and frame_idx % 30 == 0:
                percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f" ➢ Progress: Frame {frame_idx}/{total_frames} ({percent:.1f}%) | Real-time speed: {avg_fps:.1f} FPS")

    finally:
        # Release resources
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if args.show:
            cv2.destroyAllWindows()
            
    print("\n" + "=" * 60)
    print("[✔] Video/Webcam processing complete!")
    if args.save:
        print(f"[✔] Output video saved at:")
        print(f"    👉 {out_path}")
    print("=" * 60)

def main():
    args = parse_args()
    
    print("=" * 60)
    print("      PROFESSIONAL POTHOLE DETECTION INFERENCE (HUD-ON)")
    print("=" * 60)
    print(f"[*] Loading model from: {args.weights}")
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        print("Please provide the correct weights path with --weights.")
        sys.exit(1)

    try:
        # Load YOLOv8 model
        model = YOLO(args.weights)
        
        # Detect the running hardware device
        device_name = "CPU"
        if hasattr(model, "device"):
            if "cuda" in str(model.device):
                device_name = f"GPU ({torch.cuda.get_device_name(0)})" if 'torch' in sys.modules else "GPU (CUDA)"
            elif "mps" in str(model.device):
                device_name = "Apple Silicon GPU"
        # Fallback device name
        if args.device:
            device_name = f"Custom Device ({args.device})"
            
        print(f"[*] Detected hardware device: {device_name}")
        print(f"[*] Confidence threshold: {args.conf}")
        print(f"[*] IoU threshold (NMS): {args.iou}")
        print(f"[*] Inference image size: {args.imgsz}")
        print("-" * 60)
        
        source = args.source
        
        # Check whether source is webcam or file
        if source.isdigit():
            # Webcam
            process_video(model, source, args, device_name)
        else:
            if not os.path.exists(source):
                print(f"[ERROR] Input source does not exist: {source}")
                sys.exit(1)
                
            path = Path(source)
            if path.is_file():
                img_suffixes = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']
                vid_suffixes = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.3gp', '.mpeg']
                
                suffix = path.suffix.lower()
                if suffix in img_suffixes:
                    process_image(model, str(path), args, device_name)
                elif suffix in vid_suffixes:
                    process_video(model, str(path), args, device_name)
                else:
                    print(f"[WARNING] Unrecognized file extension {suffix}. Attempting to process as video...")
                    process_video(model, str(path), args, device_name)
                    
            elif path.is_dir():
                # Process a directory of images
                print(f"[*] Scanning and processing all images in directory: {source}")
                img_suffixes = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
                all_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in img_suffixes]
                
                if not all_files:
                    print(f"[WARNING] No images found in directory: {source}")
                    sys.exit(0)
                    
                print(f"[*] Found {len(all_files)} images. Starting processing...")
                for idx, file_path in enumerate(all_files, 1):
                    print(f"\n[{idx}/{len(all_files)}]")
                    process_image(model, str(file_path), args, device_name)
                    
                print("\n" + "=" * 60)
                print(f"[✔] Finished processing all images in directory!")
                print(f"[✔] All results saved to directory: {args.save_dir}")
                print("=" * 60)
                
    except Exception as e:
        print(f"[ERROR] An error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
