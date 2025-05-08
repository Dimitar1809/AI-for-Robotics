import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
import cv2
from cv_bridge import CvBridge
import torch
import torch.nn.functional as F
import os
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonFollowNode(Node):
    def __init__(self):
        super().__init__('person_follow')
        # Parameters
        self.declare_parameter('yolo_weights', 'yolov8n.pt')
        self.declare_parameter('midas_model_type', 'MiDaS_small')
        self.declare_parameter('desired_distance', 2.0)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('conf_threshold', 0.5)
        
        # Get parameter values
        yolo_weights = self.get_parameter('yolo_weights').get_parameter_value().string_value
        midas_type = self.get_parameter('midas_model_type').get_parameter_value().string_value
        self.desired_distance = self.get_parameter('desired_distance').get_parameter_value().double_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        
        # Get package share directory for weights
        pkg_dir = get_package_share_directory('relbot_video_interface')
        weights_path = os.path.join(pkg_dir, 'weights', yolo_weights)
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Load YOLOv8 model
        self.model = YOLO(weights_path)
        self.model.fuse()
        
        # Initialize DeepSORT
        self.tracker = DeepSort(max_age=30, n_init=3)
        
        # Load MiDaS depth model
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", midas_type)
        except Exception as e:
            self.get_logger().error(f"MiDaS model download/loading failed: {e}")
            raise
        
        self.midas.eval()
        
        # MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if midas_type == "DPT_Large" or midas_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas.to(self.device)
        
        # ROS publishers/subscribers
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.pos_pub = self.create_publisher(Point, '/object_position', 10)
        self.id_pub = self.create_publisher(Int32, '/object_id', 10)
        
        self.target_id = None
        self.target_depth_ref = None
        
        self.get_logger().info("PersonFollowNode initialized: YOLOv8 + DeepSORT + MiDaS for follow-me behavior.")
    
    def image_callback(self, msg: Image):
        # Convert ROS image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        img_h, img_w, _ = frame.shape
        
        # YOLO detection
        results = self.model.predict(frame, conf=0.5, iou=0.5, classes=[0])  # class 0 = person
        detections = results[0]
        
        # Prepare detections for DeepSORT
        det_list = []
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            if cls_id == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                det_list.append([[x1, y1, w, h], conf, cls_id])
        
        # Update tracker
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        
        # Find target track
        target_track = None
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            if self.target_id is not None:
                if track.track_id == self.target_id:
                    target_track = track
                    break
            else:
                target_track = track
                self.target_id = track.track_id
                break
        
        if target_track is None:
            self.target_id = None
            self.target_depth_ref = None
            return
        
        # Get target position
        x1, y1, x2, y2 = map(int, target_track.to_ltrb())
        cx = x1 + (x2 - x1)//2
        cy = y1 + (y2 - y1)//2
        
        # Compute horizontal and vertical offsets
        center_offset_x = cx - (img_w // 2)
        center_offset_y = cy - (img_h // 2)
        norm_offset_x = center_offset_x / (img_w / 2.0)
        norm_offset_y = center_offset_y / (img_h / 2.0)
        
        # Depth estimation with MiDaS
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        
        with torch.no_grad():
            depth_prediction = self.midas(input_batch)
            depth_prediction = F.interpolate(
                depth_prediction.unsqueeze(1),
                size=(img_h, img_w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        depth_map = depth_prediction.cpu().numpy()
        
        # Get depth at person's center
        person_depth_value = depth_map[cy, cx]
        
        # Initialize reference depth if not set
        if self.target_depth_ref is None:
            self.target_depth_ref = person_depth_value
            self.get_logger().info(f"Set reference depth for target ID {self.target_id}: {person_depth_value:.3f}")
        
        # Compute depth error
        depth_error = self.target_depth_ref - person_depth_value
        
        # Create and publish position message
        pos_msg = Point()
        pos_msg.x = float(norm_offset_x)   # horizontal offset
        pos_msg.y = float(norm_offset_y)   # vertical offset
        pos_msg.z = float(depth_error)     # forward distance error
        self.pos_pub.publish(pos_msg)
        
        # Publish target ID
        id_msg = Int32()
        id_msg.data = int(target_track.track_id)
        self.id_pub.publish(id_msg)
        
        # Log info
        self.get_logger().info(f"Follow ID {target_track.track_id}: depth_val={person_depth_value:.2f}, depth_err={depth_error:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown() 