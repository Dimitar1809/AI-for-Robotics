#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2
import torch
from torch.serialization import load as _orig_torch_load

def _load_no_safe(*args, **kwargs):
    kwargs.pop("weights_only", None)
    return _orig_torch_load(*args, weights_only=False, **kwargs)

torch.load = _load_no_safe

import ultralytics
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory
import gi
import numpy as np
import cv2
import os
from cv_bridge import CvBridge

# DeepSORT tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

# Torch layers whitelist
from torch.nn import Conv2d, SiLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.batchnorm import BatchNorm2d
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f

# MiDaS & transforms
import sys
sys.path.insert(0, "/home/robot/models/MiDaS")
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose 

import threading
import queue
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst

class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        # Publisher: sends object position to the RELBot
        # Topic `/object_position` is watched by the robot controller for actuation
        self.position_pub = self.create_publisher(Point, '/object_position', 10)

        # Declare GStreamer pipeline as a parameter for flexibility
        self.declare_parameter('gst_pipeline', (
            'udpsrc port=5000 caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96" ! '
            'rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw,format=RGB ! appsink name=sink'
        ))

        pipeline_str = self.get_parameter('gst_pipeline').value

        # Declare parameters
        self.declare_parameter('yolo_weights', 'best.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('MiDaS_weights', 'dpt_swin2_tiny_256.pt')

        # Get parameter values
        yolo_weights = self.get_parameter('yolo_weights').get_parameter_value().string_value
        MiDaS_weights = self.get_parameter('MiDaS_weights').get_parameter_value().string_value
        # Get package share directory for weights
        pkg_dir = get_package_share_directory('relbot_video_interface')
        weights_path = os.path.join(pkg_dir, 'weights', yolo_weights)

        self.get_logger().info(f"Loading custom YOLO model from: {weights_path}")
        self.model = YOLO(weights_path)

        # Initialize GStreamer and build pipeline
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        # Drop late frames to ensure real-time processing
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)

        # Initialize CV Bridge
        self.bridge = CvBridge()
        # Load YOLOv8 model
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel, Sequential, Conv, Conv2d, BatchNorm2d, SiLU, C2f])

        # New MiDaS loading approach
        midas_weights_filename = self.get_parameter('MiDaS_weights').get_parameter_value().string_value
        pkg_dir = get_package_share_directory('relbot_video_interface') 
        midas_weights_path = os.path.join(pkg_dir, 'weights', midas_weights_filename)

        self.get_logger().info(f"Instantiating DPTDepthModel with backbone 'swin2t16_256' for MiDaS model: {midas_weights_filename}")
        self.model_midas = DPTDepthModel(
            path=None, 
            backbone="swin2t16_256",
            non_negative=True
        )

        self.get_logger().info(f"Attempting to load state_dict for MiDaS from local file: {midas_weights_path}")
        try:
            state_dict = torch.load(midas_weights_path, map_location=torch.device('cpu')) 
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict: # Another common key
                state_dict = state_dict["model"]

            self.model_midas.load_state_dict(state_dict)
            self.get_logger().info("Successfully loaded MiDaS model weights using explicit load_state_dict.")
        except Exception as e:
            self.get_logger().error(f"Error explicitly loading MiDaS state_dict from {midas_weights_path}: {e}")
            raise 

        dpt_swin2_tiny_transform = Compose(
            [
                Resize(
                    256,  # Target width for dpt_swin2_tiny_256
                    256,  # Target height for dpt_swin2_tiny_256
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,  # Common for MiDaS DPT models, should be fine for Swin too
                    resize_method="minimal", 
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Standard DPT normalization
                PrepareForNet(),
            ]
        )
        self.transform = dpt_swin2_tiny_transform 
        self.model.fuse()  # fuse model layers for speed
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=3)

        # Add thread-safe queue and shared variables for MiDaS
        self.midas_queue = queue.Queue(maxsize=2)  # Queue for frames to process
        self.latest_depth_prediction = None
        self.latest_depth_timestamp = None  # Track when the prediction was made
        self.current_depth_prediction = None
        self.latest_depth_lock = threading.Lock()
        self.midas_running = True
        self.midas_processed_frames = 0  # Counter for logging
        self.midas_last_log_time = time.time()  # For rate logging

        # Start MiDaS processing thread
        self.midas_thread = threading.Thread(target=self._midas_processing_loop)
        self.midas_thread.daemon = True
        self.midas_thread.start()
        self.get_logger().info('MiDaS processing thread started at 1Hz')

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

        self.frame_count = 0

        script_dir = os.getcwd()
        self.video_path = os.path.join(script_dir, 'output_final.mp4')
        self.video_writer = None  # will initialize upon first frame
        self.get_logger().info(f"Will record video to: {self.video_path}")

    def _midas_processing_loop(self):
        """Background thread for MiDaS depth estimation."""
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_midas.to(device).eval()
        
        while self.midas_running:
            try:
                # Get frame from queue with timeout - 1Hz processing
                frame_rgb = self.midas_queue.get(timeout=1.0)
                
                # Process frame with MiDaS
                frame_rgb_normalized = frame_rgb.astype(np.float32) / 255.0
                midas_input_dict = {"image": frame_rgb_normalized}
                transformed_sample_dict = self.transform(midas_input_dict)
                processed_image_np = transformed_sample_dict["image"]
                input_batch = torch.from_numpy(processed_image_np).to(device).unsqueeze(0)
                
                with torch.no_grad():
                    midas_raw_prediction = self.model_midas(input_batch)
                    # Interpolate to original frame size
                    height, width = frame_rgb.shape[:2]
                    depth_prediction = torch.nn.functional.interpolate(
                        midas_raw_prediction.unsqueeze(1),
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze().cpu().numpy()
                
                # Update shared depth prediction with timestamp
                current_time = time.time()
                with self.latest_depth_lock:
                    self.latest_depth_prediction = depth_prediction
                    self.latest_depth_timestamp = current_time
                
                # Log processing rate every 10 seconds
                self.midas_processed_frames += 1
                if current_time - self.midas_last_log_time >= 10.0:
                    rate = self.midas_processed_frames / (current_time - self.midas_last_log_time)
                    self.get_logger().info(f'MiDaS processing rate: {rate:.2f} Hz')
                    self.midas_processed_frames = 0
                    self.midas_last_log_time = current_time
                
                self.midas_queue.task_done()
                
            except queue.Empty:
                # No frame to process, continue waiting
                continue
            except Exception as e:
                self.get_logger().error(f"Error in MiDaS processing thread: {e}")
                continue

    def on_timer(self):
        # Pull the latest frame from the GStreamer appsink
        sample = self.sink.emit('pull-sample')
        if not sample:
            # No new frame available
            return

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        gst_width = caps.get_structure(0).get_value('width')
        gst_height = caps.get_structure(0).get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            # Failed to map buffer data
            return

        # Convert raw buffer to numpy array [height, width, channels]
        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        frame_fake = np.array(frame)

        frame_bgr = np.frombuffer(mapinfo.data, np.uint8).reshape(gst_height, gst_width, 3)
        frame_bgr_copy = np.array(frame_bgr) 

        buf.unmap(mapinfo)

        # Display the raw input frame for debugging
        # cv2.imshow('Input Stream', frame)
        # cv2.waitKey(1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_mp4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Frame for YOLO and MiDaS (RGB)
        frame_MiDaS_rgb = cv2.cvtColor(frame_bgr_copy, cv2.COLOR_BGR2RGB)
        # Frame for displaying YOLO detections (BGR)
        frame_display_MiDaS = frame_bgr_copy.copy() 

        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, fps, (width, height)
            )
            self.get_logger().info(
                f"VideoWriter opened: {self.video_path} ({width}x{height} @ {fps} FPS)"
            )
        self.video_writer.write(frame_mp4)
        # ========================================================

        #save the frame into a file
        filename = f'frame_{self.frame_count:06d}.png'
        cv2.imwrite(filename, frame_mp4)
        self.frame_count += 1

        # Load a pretrained model on COCO datatset
        results = self.model.predict(frame_rgb, conf=0.5, iou=0.5, classes=[1])  # class 0 = person
        detections = results[0]

        # # MiDaS
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_midas.to(device).eval()

        # input_batch = self.transform(frame).to(device)
        # with torch.no_grad():
        #     prediction = self.model_midas(input_batch)
        #     prediction = torch.nn.functional.interpolate(
        #         prediction.unsqueeze(1),
        #         size=frame.shape[:2],
        #         mode="bicubic",
        #         align_corners=False,
        #     ).squeeze().cpu().numpy()

        x1 = 200 #initialise
        y1 = 0 #initialise
        w = 0 #initialise
        h = 0 #initialise

        det_list = []
        helmet_detected = False
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            if cls_id == 1:  # helmet
                print("there is a helmet!")
                helmet_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                det_list.append([[x1, y1, w, h], conf, cls_id])

                cv2.rectangle(frame_fake, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame_rgb = cv2.cvtColor(frame_fake, cv2.COLOR_BGR2RGB)
                
            else:
                print("no helmet detected")
                # x1 = 300
                # y1 = 0
                # w = 0
                # h = 0

        cv2.imshow('Output Stream', frame_rgb)
        cv2.waitKey(1)

        # Compute and publish object position
        x_position = (x1 + 0.5*w)
        y_position = (y1 + 0.5*h)
      

        # Try to add frame to MiDaS processing queue
        try:
            if not self.midas_queue.full():
                self.midas_queue.put(frame_MiDaS_rgb, block=False)
            else:
                # Clear old frames if queue is full
                try:
                    self.midas_queue.get_nowait()  # Remove one old frame
                    self.midas_queue.put(frame_MiDaS_rgb, block=False)  # Add new frame
                except queue.Empty:
                    pass  
        except queue.Full:
            pass

        # Get latest depth prediction and check its age
        prediction_age = float('inf')
        with self.latest_depth_lock:
            self.current_depth_prediction = self.latest_depth_prediction
            if self.latest_depth_timestamp is not None:
                prediction_age = time.time() - self.latest_depth_timestamp

        # Log if prediction is getting old
        if prediction_age > 2.0:  # More than 2 seconds old
            self.get_logger().warn(f'MiDaS depth prediction is {prediction_age:.1f} seconds old')

        # Display depth map if available
        if self.current_depth_prediction is not None:
            depth_display_normalized = cv2.normalize(self.current_depth_prediction, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_display_colored = cv2.applyColorMap(depth_display_normalized, cv2.COLORMAP_INFERNO)
            # cv2.imshow('MiDaS Depth Map', depth_display_colored)
            
        
        # cv2.waitKey(1)

        # Use center of YOLO bounding box
        center_x_obj_pixels = float(x_position)
        center_y_obj_pixels = float(y_position)
        self.get_logger().info(f"Width ({w:.1f}), x-pos {center_x_obj_pixels:.4f}")

        object_depth_value = 1500.0  # Default value
        if helmet_detected and self.current_depth_prediction is not None:
            idx_y = np.clip(int(center_y_obj_pixels), 0, self.current_depth_prediction.shape[0] - 1)
            idx_x = np.clip(int(center_x_obj_pixels), 0, self.current_depth_prediction.shape[1] - 1)
            
            try:
                object_depth_value = float(self.current_depth_prediction[idx_y, idx_x])
                self.get_logger().info(f'MiDaS depth value is {object_depth_value:.1f}')
                if not np.isfinite(object_depth_value):  # Check for NaN or inf
                    object_depth_value = 1500.0
            except (TypeError, ValueError) as e:
                self.get_logger().warn(f"Invalid depth value: {e}, using default")
                object_depth_value = 1500.0

        # Ensure all values are valid floats before creating message
        try:
            msg = Point()
            msg.y = 0.0  # y-coordinate is always 0.0
            
            if helmet_detected:
                if center_x_obj_pixels > 160.0 and center_x_obj_pixels < 240.0:
                    center_x_obj_pixels = 200.0
                msg.x = float(center_x_obj_pixels)  
                if float(object_depth_value) < 1400.0: 
                    msg.z = float(object_depth_value)
                else:
                    msg.z = 10001.0
            else:
                msg.z = 10001.0
                msg.x = 230.0
                
            # Final validation before publishing
            if not all(np.isfinite([msg.x, msg.y, msg.z])):
                self.get_logger().error("Invalid values in Point message, skipping publish")
                return
                
            self.position_pub.publish(msg)
            
        except (TypeError, ValueError) as e:
            self.get_logger().error(f"Error creating Point message: {e}")
            return

    def destroy_node(self):
        # Signal MiDaS thread to stop
        self.midas_running = False
        if self.midas_thread.is_alive():
            self.midas_thread.join(timeout=1.0)
        
        self.pipeline.set_state(Gst.State.NULL)
        if getattr(self, 'video_writer', None) is not None:
            self.video_writer.release()
            self.get_logger().info('Video file finalized and closed.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoInterfaceNode()
    try:
        rclpy.spin(node)  # Keep node alive, invoking on_timer periodically
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()