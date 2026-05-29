import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

class CameraCalibration:
    def __init__(self, camera_height=1.5, pitch_deg=15, fov_deg=60, img_width=1280, img_height=720):
        self.h = camera_height
        self.pitch = np.radians(pitch_deg)
        self.fov = np.radians(fov_deg)
        self.W = img_width
        self.H = img_height
        
        # Estimate focal length from FOV
        self.f_x = self.W / (2 * np.tan(self.fov / 2))
        self.f_y = self.f_x
        self.c_x = self.W / 2
        self.c_y = self.H / 2
        
        # Intrinsic matrix
        self.K = np.array([
            [self.f_x, 0, self.c_x],
            [0, self.f_y, self.c_y],
            [0, 0, 1]
        ], dtype=np.float32)

    def image_to_ground(self, u, v):
        """
        Convert pixel coordinate (u, v) to ground coordinates (X, Y) 
        assuming flat ground and given camera height & pitch.
        Z is along the ground plane forward, X is right.
        """
        # Pixel to normalized ray
        x_norm = (u - self.c_x) / self.f_x
        y_norm = (v - self.c_y) / self.f_y
        
        # Camera rotation for pitch (rotation around Xc axis)
        c_p = np.cos(self.pitch)
        s_p = np.sin(self.pitch)
        
        # Ground intersection constraint
        denom = y_norm * c_p + s_p
        if denom <= 1e-6:
            # Ray points above or parallel to the horizon
            return None, None
            
        lam = self.h / denom
        Xc = lam * x_norm
        Yc = lam * y_norm
        Zc = lam * 1.0
        
        # Transform to ground frame:
        # X_world = Xc
        # Z_world = -Yc * s_p + Zc * c_p
        X_world = Xc
        Z_world = -Yc * s_p + Zc * c_p
        
        return X_world, Z_world

    def get_area(self, box):
        """ Calculate physical area (m^2) using robust Center-Projected IPM """
        x1, y1, x2, y2 = box
        u_center = (x1 + x2) / 2.0
        v_center = (y1 + y2) / 2.0
        
        # Project left and right centers to get physical width
        X_left, _ = self.image_to_ground(x1, v_center)
        X_right, _ = self.image_to_ground(x2, v_center)
        
        # Project top and bottom centers to get physical length
        _, Z_top = self.image_to_ground(u_center, y1)
        _, Z_bottom = self.image_to_ground(u_center, y2)
        
        if any(v is None for v in [X_left, X_right, Z_top, Z_bottom]):
            return 0.0
            
        W = abs(X_right - X_left)
        L = abs(Z_top - Z_bottom)
        
        # Area of ellipse (closer representation of pothole shape)
        return (np.pi / 4.0) * W * L

    def get_distance(self, box):
        """ Get ground distance (m) to the bottom center of the pothole """
        x1, y1, x2, y2 = box
        u_center = (x1 + x2) / 2.0
        _, Z_bottom = self.image_to_ground(u_center, y2)
        if Z_bottom is None:
            return 0.0
        return Z_bottom


class DepthAreaEstimator:
    def __init__(self, yolo_path, depth_onnx_path, device='cpu', camera_height=1.5, pitch_deg=15, fov_deg=60, depth_size=266, depth_scale=0.015):
        self.device = device
        self.camera_height = camera_height
        self.pitch_deg = pitch_deg
        self.fov_deg = fov_deg
        self.depth_size = depth_size
        self.depth_scale = depth_scale
        
        # Load YOLO
        self.yolo = YOLO(yolo_path)
        
        # Load Depth Anything V2 ONNX
        available_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # CPU Multi-threading Optimization: apply if device is CPU or if CUDA was requested but is not available on this machine
        actual_use_cpu = ('cpu' in device.lower()) or ('CUDAExecutionProvider' not in available_providers)
        if actual_use_cpu:
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 1
            
        self.depth_session = ort.InferenceSession(depth_onnx_path, sess_options=sess_options, providers=providers)
        
        # Default camera params will be set when first frame arrives to match resolution
        self.cam = None

    def run_depth_anything(self, image):
        """ Run Depth Anything V2 with optimized dynamic size """
        h, w = image.shape[:2]
        
        # Preprocessing: resize to dynamic size (must be a multiple of 14)
        input_size = self.depth_size
        img_resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0
        img_norm = (img_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_input = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        # Inference
        ort_inputs = {self.depth_session.get_inputs()[0].name: img_input}
        depth_map = self.depth_session.run(None, ort_inputs)[0]
        
        # Postprocessing
        depth_map = depth_map.squeeze()
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth_map

    def run_yolo(self, image):
        """ Run YOLOv8 """
        results = self.yolo.predict(image, conf=0.2, iou=0.45, verbose=False)
        return results[0]

    def estimate_pothole_depth(self, depth_map, box, z_road):
        """
        Estimate physical depth using robust relative disparity method.
        Depth Anything V2 outputs disparity-like values (d ~ 1/Z).
        Physical depth delta_Z = Z_road * (d_road - d_pothole) / d_pothole.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = depth_map.shape
        
        # Bounding box ROI
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        roi_depth = depth_map[y1:y2, x1:x2]
        
        # Potholes are indentations, meaning they are further away than the surrounding road.
        # In disparity maps, smaller values = further away.
        # d_road: estimate from the boundary of the bounding box
        top_edge = roi_depth[0, :]
        bottom_edge = roi_depth[-1, :]
        left_edge = roi_depth[:, 0]
        right_edge = roi_depth[:, -1]
        
        boundary_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
        if len(boundary_pixels) == 0:
            return 0.0
        d_road = np.percentile(boundary_pixels, 50) # median of boundary
        
        # d_pothole: estimate from the deepest part (minimum disparity)
        d_pothole = np.percentile(roi_depth, 10) # 10th percentile to avoid noise
        
        if d_pothole >= d_road or d_pothole <= 0:
            return 0.0 # Not a pothole (or noisy)
            
        # Delta Z (m) = Z_road * (d_road - d_pothole) / d_pothole
        # Multiply by a calibration scale factor.
        depth_m = z_road * ((d_road - d_pothole) / d_pothole) * (self.depth_scale * 10)
        depth_cm = depth_m * 100.0
        
        # Safeguard: clip depth to physically realistic limits (e.g. 0 to 30 cm)
        depth_cm = np.clip(depth_cm, 0.0, 30.0)
        
        return float(depth_cm)

    def process_frame(self, frame):
        """ Optimized pipeline: Runs YOLO first, only runs Depth if potholes are detected """
        if self.cam is None:
            self.cam = CameraCalibration(
                camera_height=self.camera_height, 
                pitch_deg=self.pitch_deg, 
                fov_deg=self.fov_deg, 
                img_width=frame.shape[1], 
                img_height=frame.shape[0]
            )

        # Run YOLOv8 first (very fast on CPU)
        yolo_result = self.run_yolo(frame)
        boxes = yolo_result.boxes
        
        results_out = []
        depth_map = None
        
        # Conditional execution of Depth Anything V2
        if len(boxes) > 0:
            # Only run Depth model when there is a pothole detected
            depth_map = self.run_depth_anything(frame)
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # Ground distance estimation
                z_road = self.cam.get_distance((x1, y1, x2, y2))
                
                # Area estimation (m^2)
                area_m2 = self.cam.get_area((x1, y1, x2, y2))
                
                # Depth estimation (cm)
                depth_cm = self.estimate_pothole_depth(depth_map, (x1, y1, x2, y2), z_road)
                
                # Severity classification
                severity = "minor"
                if area_m2 > 0.5 or depth_cm > 10.0:
                    severity = "severe"
                elif area_m2 > 0.2 or depth_cm > 5.0:
                    severity = "moderate"
                    
                results_out.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "area_m2": area_m2,
                    "depth_cm": depth_cm,
                    "distance_m": z_road,
                    "severity": severity
                })
                
        return results_out, depth_map
