# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Point
# import gi
# import numpy as np
# import cv2
# import ultralytics  # Assuming ultralytics is installed for YOLOv8
# from ultralytics import YOLO
# import torch
# from cv_bridge import CvBridge
# import os
# from ament_index_python.packages import get_package_share_directory
# from deep_sort_realtime.deepsort_tracker import DeepSort

# gi.require_version('Gst', '1.0')
# from gi.repository import Gst

# class VideoInterfaceNode(Node):
#     def __init__(self):
#         super().__init__('video_interface')
#         # Publisher: sends object position to the RELBot
#         # Topic `/object_position` is watched by the robot controller for actuation
#         self.position_pub = self.create_publisher(Point, '/object_position', 10)

#         # Declare GStreamer pipeline as a parameter for flexibility
#         self.declare_parameter('gst_pipeline', (
#             'udpsrc port=5000 caps="application/x-rtp,media=video,'
#             'encoding-name=H264,payload=96" ! '
#             'rtph264depay ! avdec_h264 ! videoconvert ! '
#             'video/x-raw,format=RGB ! appsink name=sink'
#         ))
#         pipeline_str = self.get_parameter('gst_pipeline').value

#         # Declare parameters
#         self.declare_parameter('yolo_weights', 'best.pt')
#         self.declare_parameter('conf_threshold', 0.5)

#         # Get parameter values
#         yolo_weights = self.get_parameter('yolo_weights').get_parameter_value().string_value

#         # Get package share directory for weights
#         pkg_dir = get_package_share_directory('relbot_video_interface')
#         weights_path = os.path.join(pkg_dir, 'weights', yolo_weights)

#         # Initialize GStreamer and build pipeline
#         Gst.init(None)
#         self.pipeline = Gst.parse_launch(pipeline_str)
#         self.sink = self.pipeline.get_by_name('sink')
#         # Drop late frames to ensure real-time processing
#         self.sink.set_property('drop', True)
#         self.sink.set_property('max-buffers', 1)
#         self.pipeline.set_state(Gst.State.PLAYING)

#         # Initialize CV Bridge
#         self.bridge = CvBridge()
#         # Load YOLOv8 model
#         self.model = YOLO(weights_path)
#         self.model.fuse()  # fuse model layers for speed
#         # Initialize DeepSORT tracker
#         self.tracker = DeepSort(max_age=30, n_init=3)

#         # Timer: fires at ~30Hz to pull frames and publish positions
#         # The period (1/30) sets how often on_timer() is called
#         self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
#         self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

#         # for saving the frame
#         self.frame_count = 0

#     def on_timer(self):
#         # Pull the latest frame from the GStreamer appsink
#         sample = self.sink.emit('pull-sample')
#         if not sample:
#             # No new frame available
#             return

#         buf = sample.get_buffer()
#         caps = sample.get_caps()
#         width = caps.get_structure(0).get_value('width')
#         height = caps.get_structure(0).get_value('height')
#         ok, mapinfo = buf.map(Gst.MapFlags.READ)
#         if not ok:
#             # Failed to map buffer data
#             return

#         # Convert raw buffer to numpy array [height, width, channels]
#         frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
#         frame_fake = np.array(frame)
#         buf.unmap(mapinfo)

#         # Display the raw input frame for debugging
#         cv2.imshow('Input Stream', frame)
#         cv2.waitKey(1)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_mp4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         #save the frame into a file
#         filename = f'frame_{self.frame_count:06d}.png'
#         cv2.imwrite(filename, frame_mp4)
#         self.frame_count += 1

#         # Load a pretrained model on COCO datatset
#         results = self.model.predict(frame_rgb, conf=0.5, iou=0.5, classes=[1])  # class 0 = person
#         detections = results[0]

#         x1 = 200 #initialise
#         w = 0 #initialise
#         h = 0 #initialise

#         # Prepare detections in DeepSORT format: [[xmin,ymin,width,height], confidence, class_id]
#         det_list = []
#         for box in detections.boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             if conf < 0.5:
#                 continue
#             if cls_id == 1:  # helmet
#                 print("there is a helmet!")
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 w = x2 - x1
#                 h = y2 - y1
#                 det_list.append([[x1, y1, w, h], conf, cls_id])

#                 cv2.rectangle(frame_fake, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 frame_rgb = cv2.cvtColor(frame_fake, cv2.COLOR_BGR2RGB)
                
#             else:
#                 print("no helmet detected")
#                 x1 = 200
#                 w = 0
#                 h = 0

#         # Display the raw input frame for debugging

#         cv2.imshow('Output Stream', frame_rgb)
#         cv2.waitKey(1)

#         # h, w, c = frame.shape
#         # print("height", h)
#         # print("width", w)

#         # TODO: Insert detection/tracking logic here to compute object position
#         # For demonstration, here we are publishing a dummy Point at origin
#                 # Compute and publish object position:
#         # x = horizontal center coordinate of the object
#         x_position = (x1 + 0.5*w)

#         # y = unused (flat-ground assumption)
#         # z = object area (controller caps at 10000 to stop robot when object is too large)
#         area = w*h
#         print("Area:", area)
    
#         msg = Point()
#         msg.x = x_position  # object center x-coordinate
#         if area < 1350:
#             msg.z = 10001.0
#         else: 
#             msg.z = float(area)
#         # msg.x = 1.0  # object center x-coordinate
#         msg.y = 0.0  # y-coordinate unused
#         # msg.z = 10001.0  # object area; >10000 indicates 'too close'
#         # msg.z = 10001.0  # object area; >10000 indicates 'too close'
#         self.position_pub.publish(msg)
#         # To adjust robot behavior, apply a scaling factor to 'z' (e.g., couple with depth estimation)
#         # Log at debug level if needed:
#         # self.get_logger().debug(f'Published position: ({msg.x}, {msg.y}, {msg.z})')

#     def destroy_node(self):
#         # Cleanup GStreamer resources on shutdown
#         self.pipeline.set_state(Gst.State.NULL)
#         super().destroy_node()


# def main(args=None):
#     rclpy.init(args=args)
#     node = VideoInterfaceNode()
#     try:
#         rclpy.spin(node)  # Keep node alive, invoking on_timer periodically
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# ## This one has the video 
# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Point
# import gi
# import numpy as np
# import cv2
# import ultralytics  # Assuming ultralytics is installed for YOLOv8
# from ultralytics import YOLO
# import torch
# from cv_bridge import CvBridge
# import os
# from ament_index_python.packages import get_package_share_directory
# from deep_sort_realtime.deepsort_tracker import DeepSort

# gi.require_version('Gst', '1.0')
# from gi.repository import Gst

# class VideoInterfaceNode(Node):
#     def __init__(self):
#         super().__init__('video_interface')
#         # Publisher: sends object position to the RELBot
#         # Topic `/object_position` is watched by the robot controller for actuation
#         self.position_pub = self.create_publisher(Point, '/object_position', 10)

#         # Declare GStreamer pipeline as a parameter for flexibility
#         self.declare_parameter('gst_pipeline', (
#             'udpsrc port=5000 caps="application/x-rtp,media=video,'
#             'encoding-name=H264,payload=96" ! '
#             'rtph264depay ! avdec_h264 ! videoconvert ! '
#             'video/x-raw,format=RGB ! appsink name=sink'
#         ))
#         pipeline_str = self.get_parameter('gst_pipeline').value

#         # Declare parameters
#         self.declare_parameter('yolo_weights', 'best.pt')
#         self.declare_parameter('conf_threshold', 0.5)

#         # Get parameter values
#         yolo_weights = self.get_parameter('yolo_weights').get_parameter_value().string_value

#         # Get package share directory for weights
#         pkg_dir = get_package_share_directory('relbot_video_interface')
#         weights_path = os.path.join(pkg_dir, 'weights', yolo_weights)

#         # Initialize GStreamer and build pipeline
#         Gst.init(None)
#         self.pipeline = Gst.parse_launch(pipeline_str)
#         self.sink = self.pipeline.get_by_name('sink')
#         # Drop late frames to ensure real-time processing
#         self.sink.set_property('drop', True)
#         self.sink.set_property('max-buffers', 1)
#         self.pipeline.set_state(Gst.State.PLAYING)

#         # Initialize CV Bridge
#         self.bridge = CvBridge()
#         # Load YOLOv8 model
#         self.model = YOLO(weights_path)
#         self.model.fuse()  # fuse model layers for speed
#         # Initialize DeepSORT tracker
#         self.tracker = DeepSort(max_age=30, n_init=3)

#         # Timer: fires at ~30Hz to pull frames and publish positions
#         # The period (1/30) sets how often on_timer() is called
#         self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
#         self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

#         # for saving the frame
#         self.frame_count = 0

#         # === ADDED: setup VideoWriter ===
#         # Output video path next to this script:
#         script_dir = os.getcwd()
#         self.video_path = os.path.join(script_dir, 'output_3.mp4')
#         self.video_writer = None  # will initialize upon first frame
#         self.get_logger().info(f"Will record video to: {self.video_path}")
#         # ==================================

#     def on_timer(self):
#         # Pull the latest frame from the GStreamer appsink
#         sample = self.sink.emit('pull-sample')
#         if not sample:
#             # No new frame available
#             return

#         buf = sample.get_buffer()
#         caps = sample.get_caps()
#         width = caps.get_structure(0).get_value('width')
#         height = caps.get_structure(0).get_value('height')
#         ok, mapinfo = buf.map(Gst.MapFlags.READ)
#         if not ok:
#             # Failed to map buffer data
#             return

#         # Convert raw buffer to numpy array [height, width, channels]
#         frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
#         frame_fake = np.array(frame)
#         buf.unmap(mapinfo)

#         # Display the raw input frame for debugging
#         cv2.imshow('Input Stream', frame)
#         cv2.waitKey(1)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_mp4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # === ADDED: initialize writer on first frame & write ===
#         if self.video_writer is None:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = 30.0
#             self.video_writer = cv2.VideoWriter(
#                 self.video_path, fourcc, fps, (width, height)
#             )
#             self.get_logger().info(
#                 f"VideoWriter opened: {self.video_path} ({width}x{height} @ {fps} FPS)"
#             )
#         self.video_writer.write(frame_mp4)
#         # ========================================================

#         #save the frame into a file
#         filename = f'frame_{self.frame_count:06d}.png'
#         cv2.imwrite(filename, frame_mp4)
#         self.frame_count += 1

#         # Load a pretrained model on COCO datatset
#         results = self.model.predict(frame_rgb, conf=0.5, iou=0.5, classes=[1])  # class 0 = person
#         detections = results[0]

#         x1 = 200 #initialise
#         w = 0 #initialise
#         h = 0 #initialise

#         # Prepare detections in DeepSORT format: [[xmin,ymin,width,height], confidence, class_id]
#         det_list = []
#         for box in detections.boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             if conf < 0.5:
#                 continue
#             if cls_id == 1:  # helmet
#                 print("there is a helmet!")
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 w = x2 - x1
#                 h = y2 - y1
#                 det_list.append([[x1, y1, w, h], conf, cls_id])

#                 cv2.rectangle(frame_fake, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 frame_rgb = cv2.cvtColor(frame_fake, cv2.COLOR_BGR2RGB)
                
#             else:
#                 print("no helmet detected")
#                 x1 = 200
#                 w = 0
#                 h = 0

#         # Display the raw input frame for debugging
#         cv2.imshow('Output Stream', frame_rgb)
#         cv2.waitKey(1)

#         # Compute and publish object position
#         x_position = (x1 + 0.5*w)
#         area = w*h
#         print("Area:", area)
    
#         msg = Point()
#         msg.x = x_position  # object center x-coordinate
#         if area < 1500:
#             msg.z = 10001.0
#         else: 
#             msg.z = float(area)
#         msg.y = 0.0  # y-coordinate unused
#         self.position_pub.publish(msg)

#     def destroy_node(self):
#         # Cleanup GStreamer resources on shutdown
#         self.pipeline.set_state(Gst.State.NULL)
#         # === ADDED: release VideoWriter ===
#         if getattr(self, 'video_writer', None) is not None:
#             self.video_writer.release()
#             self.get_logger().info('Video file finalized and closed.')
#         # ======================================
#         super().destroy_node()


# def main(args=None):
#     rclpy.init(args=args)
#     node = VideoInterfaceNode()
#     try:
#         rclpy.spin(node)  # Keep node alive, invoking on_timer periodically
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

## Midas
#!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Point
# import gi
# import numpy as np
# import cv2
# import torch
# from torch.serialization import load as _orig_torch_load
# _torch_load = torch.load
# def _load_no_safe(*args, **kwargs):
#     return _torch_load(*args, weights_only=False, **kwargs)
# torch.load = _load_no_safe
# import ultralytics  # Assuming ultralytics is installed for YOLOv8
# from ultralytics import YOLO
# from torch.nn import Conv2d
# import ultralytics
# from ultralytics.nn.modules.conv import Conv
# from ultralytics.nn.modules.block import C2f
# import torch.nn.modules.container
# from torch.nn.modules.container import Sequential
# from torch.nn.modules.batchnorm import BatchNorm2d
# from torch.nn import SiLU
# from cv_bridge import CvBridge
# import os
# from ament_index_python.packages import get_package_share_directory
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import sys 
# # Import the model definitions and transforms
# sys.path.insert(0, "/home/robot/models/MiDaS")
# from midas.dpt_depth import DPTDepthModel
# from midas.midas_net import MidasNet
# from midas.transforms import Resize, NormalizeImage, PrepareForNet
# from torchvision.transforms import Compose

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
# from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose 

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
        self.declare_parameter('MiDaS_weights', 'dpt_swin2_tiny_256.pt')     # <<< New default

        # Get parameter values
        yolo_weights = self.get_parameter('yolo_weights').get_parameter_value().string_value
        MiDaS_weights = self.get_parameter('MiDaS_weights').get_parameter_value().string_value
        # Get package share directory for weights
        pkg_dir = get_package_share_directory('relbot_video_interface')
        weights_path = os.path.join(pkg_dir, 'weights', yolo_weights)\


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
            # Use your _load_no_safe function which wraps torch.load with weights_only=False
            # This _torch_load is the one you defined at the top of your script.
            state_dict = torch.load(midas_weights_path, map_location=torch.device('cpu')) 

            # Some checkpoints might have the state_dict nested, common in training checkpoints
            # but official release .pt files from MiDaS are usually direct state_dicts.
            # We can add a check just in case, though it might not be needed for official .pt files.
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict: # Another common key
                state_dict = state_dict["model"]

            self.model_midas.load_state_dict(state_dict)
            self.get_logger().info("Successfully loaded MiDaS model weights using explicit load_state_dict.")
        except Exception as e:
            self.get_logger().error(f"Error explicitly loading MiDaS state_dict from {midas_weights_path}: {e}")
            raise # Re-raise the exception to see the full traceback

        dpt_swin2_tiny_transform = Compose(
            [
                Resize(
                    256,  # Target width for dpt_swin2_tiny_256
                    256,  # Target height for dpt_swin2_tiny_256
                    resize_target=None,
                    keep_aspect_ratio=True,
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

     
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

     
        self.frame_count = 0

        script_dir = os.getcwd()
        self.video_path = os.path.join(script_dir, 'output_3.mp4')
        self.video_writer = None  # will initialize upon first frame
        self.get_logger().info(f"Will record video to: {self.video_path}")

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
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            # Failed to map buffer data
            return

        # Convert raw buffer to numpy array [height, width, channels]
        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        frame_fake = np.array(frame)
        buf.unmap(mapinfo)

        # Display the raw input frame for debugging
        cv2.imshow('Input Stream', frame)
        cv2.waitKey(1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_mp4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        # MiDaS
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_midas.to(device).eval()

        input_batch = self.transform(frame).to(device)
        with torch.no_grad():
            prediction = self.model_midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()


        

        x1 = 200 #initialise
        w = 0 #initialise
        h = 0 #initialise

        det_list = []
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            if cls_id == 1:  # helmet
                print("there is a helmet!")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                det_list.append([[x1, y1, w, h], conf, cls_id])

                cv2.rectangle(frame_fake, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame_rgb = cv2.cvtColor(frame_fake, cv2.COLOR_BGR2RGB)
                
            else:
                print("no helmet detected")
                x1 = 200
                w = 0
                h = 0

        cv2.imshow('Output Stream', frame_rgb)
        cv2.waitKey(1)

        # Compute and publish object position
        x_position = (x1 + 0.5*w)
        y_position = (y1 + 0.5*h)

        print(f"Relative depth at ({x_position},{y_position}): {prediction[y_position,x_position]:.4f}")
        area = w*h
        print("Area:", area)
    
        msg = Point()
        msg.x = x_position  # object center x-coordinate
        if area < 1500:
            msg.z = 10001.0
        else: 
            msg.z = float(area)
        msg.y = 0.0  # y-coordinate 
        self.position_pub.publish(msg)

    def destroy_node(self):
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
