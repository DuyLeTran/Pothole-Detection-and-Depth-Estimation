import cv2
import time
import argparse
import os
import sys
from depth_area_estimator import DepthAreaEstimator

def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Pipeline for Pothole Detection & Depth/Area Estimation")
    parser.add_argument("--source", type=str, required=True, help="Path to input video or webcam index")
    parser.add_argument("--yolo-weights", type=str, default="checkpoints/yolov8n_best.onnx", help="Path to YOLOv8 weights")
    parser.add_argument("--depth-weights", type=str, default="checkpoints/depth_anything_v2_vits_dynamic.onnx", help="Path to Depth Anything V2 ONNX")
    parser.add_argument("--save-dir", type=str, default="runs/inference_pipeline", help="Dir to save output")
    
    # Camera calibration parameters
    parser.add_argument("--camera-height", type=float, default=1.5, help="Camera mounting height in meters (default: 1.5)")
    parser.add_argument("--pitch", type=float, default=15.0, help="Camera pitch angle downwards in degrees (default: 15.0)")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera Field of View in degrees (default: 60.0)")
    
    # Depth Anything V2 parameters
    parser.add_argument("--depth-size", type=int, default=266, help="Input size for Depth Anything V2 (default: 266)")
    parser.add_argument("--depth-scale", type=float, default=0.015, help="Scaling factor for relative-to-physical depth (default: 0.015)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the pipeline on (cuda or cpu)")
    parser.add_argument("--show", action="store_true", help="Display the output window in real-time")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize Estimator
    print("[*] Loading Models...")
    estimator_kwargs = {
        "yolo_path": args.yolo_weights,
        "depth_onnx_path": args.depth_weights,
        "camera_height": args.camera_height,
        "pitch_deg": args.pitch,
        "fov_deg": args.fov,
        "depth_size": args.depth_size,
        "depth_scale": args.depth_scale
    }
    
    try:
        estimator = DepthAreaEstimator(device=args.device, **estimator_kwargs)
    except Exception as e:
        print(f"[!] Warning: CUDA failed, falling back to CPU. Error: {e}")
        estimator = DepthAreaEstimator(device='cpu', **estimator_kwargs)

    # Open Source
    source = args.source
    is_webcam = source.isdigit()
    
    # Check if input is a static image
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    is_image = not is_webcam and source.lower().endswith(image_exts)
    
    os.makedirs(args.save_dir, exist_ok=True)
    out_name = f"pipeline_output_{int(time.time())}.mp4" if is_webcam else f"pipeline_{os.path.basename(source)}"
    out_path = os.path.join(args.save_dir, out_name)

    if is_image:
        print(f"[*] Reading static image: {source}")
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Cannot read image: {source}")
            sys.exit(1)
            
        height, width = frame.shape[:2]
        print(f"[*] Starting inference on image...")
        print(f"[*] Camera Config: Height={args.camera_height}m, Pitch={args.pitch}°, FOV={args.fov}°")
        print(f"[*] Depth model input size: {args.depth_size}x{args.depth_size}")
        
        start_t = time.time()
        results, depth_map = estimator.process_frame(frame)
        exec_time = time.time() - start_t
        print(f" ➢ Processing finished in {exec_time * 1000:.1f} ms")
        
        # Draw HUD & Results
        for res in results:
            x1, y1, x2, y2 = res['box']
            severity = res['severity']
            
            if severity == "severe":
                color = (0, 0, 255) # Red
            elif severity == "moderate":
                color = (0, 165, 255) # Orange
            else:
                color = (0, 255, 255) # Yellow
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label1 = f"{severity.upper()} ({res['conf']:.2f})"
            label2 = f"Area: {res['area_m2']:.2f} m2 | Dist: {res['distance_m']:.1f} m"
            label3 = f"Depth: {res['depth_cm']:.1f} cm"
            
            cv2.putText(frame, label1, (x1, max(0, y1 - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, label2, (x1, max(0, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, label3, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"Time: {exec_time*1000:.1f}ms", (width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Potholes: {len(results)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Optional: Render Depth Map as Picture-in-Picture
        if depth_map is not None:
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            dh, dw = int(height * 0.2), int(width * 0.2)
            depth_small = cv2.resize(depth_color, (dw, dh))
            frame[height-dh-10:height-10, width-dw-10:width-10] = depth_small
            cv2.rectangle(frame, (width-dw-10, height-dh-10), (width-10, height-10), (255,255,255), 1)
            
        cv2.imwrite(out_path, frame)
        print(f"\n[✔] Inference complete! Output saved to: {out_path}")
        
        if args.show:
            cv2.imshow("Pothole Detection & Depth/Area Pipeline", frame)
            print("[*] Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    else:
        cap_source = int(source) if is_webcam else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {source}")
            sys.exit(1)

        # Video Writer setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Try using H.264 (avc1) for web and VS Code compatibility, fallback to mp4v if unsupported
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
            if not out_writer.isOpened():
                raise ValueError("avc1 codec not supported by this OpenCV build")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

        print(f"[*] Starting inference... Press 'q' to stop.")
        print(f"[*] Camera Config: Height={args.camera_height}m, Pitch={args.pitch}°, FOV={args.fov}°")
        print(f"[*] Depth model input size: {args.depth_size}x{args.depth_size}")
        avg_fps = 0.0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            start_t = time.time()
            
            # Parallel processing
            results, depth_map = estimator.process_frame(frame)
            
            # FPS calc (skip first 5 frames to avoid warmup delay in EMA)
            exec_time = time.time() - start_t
            instant_fps = 1.0 / exec_time if exec_time > 0 else 30.0
            if frame_idx > 5:
                if avg_fps == 0.0:
                    avg_fps = instant_fps
                else:
                    avg_fps = 0.9 * avg_fps + 0.1 * instant_fps
            else:
                avg_fps = instant_fps

            # Draw HUD & Results
            for res in results:
                x1, y1, x2, y2 = res['box']
                severity = res['severity']
                
                # Colors based on severity
                if severity == "severe":
                    color = (0, 0, 255) # Red
                elif severity == "moderate":
                    color = (0, 165, 255) # Orange
                else:
                    color = (0, 255, 255) # Yellow
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Labels
                label1 = f"{severity.upper()} ({res['conf']:.2f})"
                label2 = f"Area: {res['area_m2']:.2f} m2 | Dist: {res['distance_m']:.1f} m"
                label3 = f"Depth: {res['depth_cm']:.1f} cm"
                
                cv2.putText(frame, label1, (x1, max(0, y1 - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label2, (x1, max(0, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, label3, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # FPS counter
            fps_color = (0, 255, 0) if avg_fps >= 15 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 2)
            cv2.putText(frame, f"Potholes: {len(results)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Optional: Render Depth Map as Picture-in-Picture
            if depth_map is not None:
                depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                dh, dw = int(height * 0.2), int(width * 0.2)
                depth_small = cv2.resize(depth_color, (dw, dh))
                frame[height-dh-10:height-10, width-dw-10:width-10] = depth_small
                cv2.rectangle(frame, (width-dw-10, height-dh-10), (width-10, height-10), (255,255,255), 1)

            # Save
            out_writer.write(frame)
            if frame_idx % 10 == 0:
                print(f" ➢ Processed frame {frame_idx}, Speed: {avg_fps:.1f} FPS")
                
            if args.show:
                cv2.imshow("Pothole Detection & Depth/Area Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] Exiting loop early due to 'q' press...")
                    break

        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        print(f"\n[✔] Inference complete! Output saved to: {out_path}")

if __name__ == "__main__":
    main()
