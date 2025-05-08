import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
import cv2
from cv_bridge import CvBridge
import os
from ament_index_python.packages import get_package_share_directory

# Import YOLOv8 and DeepSORT utilities
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__('person_tracker')
        # Declare parameters
        self.declare_parameter('yolo_weights', 'yolov8n.pt')  # path to YOLOv8 weights
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        
        # Get parameter values
        yolo_weights = self.get_parameter('yolo_weights').get_parameter_value().string_value
        conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        # Get package share directory for weights
        pkg_dir = get_package_share_directory('relbot_video_interface')
        weights_path = os.path.join(pkg_dir, 'weights', yolo_weights)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        # Load YOLOv8 model
        self.model = YOLO(weights_path)
        self.model.fuse()  # fuse model layers for speed
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=3)
        
        # Subscribers and Publishers
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.pos_pub = self.create_publisher(Point, '/object_position', 10)
        self.id_pub = self.create_publisher(Int32, '/object_id', 10)
        
        # Variables to keep state
        self.target_id = None  # track ID of the selected person to follow

        self.get_logger().info("PersonTrackerNode initialized: using YOLOv8 for person detection.")

    def image_callback(self, msg: Image):
        # Convert ROS Image to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return
        
        # Run YOLOv8 model inference
        results = self.model.predict(frame, conf=0.5, iou=0.5, classes=[0])  # class 0 = person
        detections = results[0]
        
        # Prepare detections in DeepSORT format: [[xmin,ymin,width,height], confidence, class_id]
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
        
        # Update tracker with current detections
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        
        # Choose target track
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
            return
        
        # Get target position and compute offset
        x1, y1, x2, y2 = map(int, target_track.to_ltrb())
        cx = x1 + (x2 - x1)//2
        cy = y1 + (y2 - y1)//2
        
        # Compute horizontal offset from image center
        img_h, img_w, _ = frame.shape
        center_offset_x = cx - (img_w // 2)
        center_offset_y = cy - (img_h // 2)
        
        # Normalize offsets
        norm_offset_x = center_offset_x / (img_w / 2.0)
        norm_offset_y = center_offset_y / (img_h / 2.0)
        
        # Create and publish position message
        pos_msg = Point()
        pos_msg.x = float(norm_offset_x)   # horizontal offset ([-1,1])
        pos_msg.y = float(norm_offset_y)   # vertical offset ([-1,1])
        pos_msg.z = 0.0  # no forward offset in 1.1
        self.pos_pub.publish(pos_msg)
        
        # Publish target ID
        id_msg = Int32()
        id_msg.data = int(target_track.track_id)
        self.id_pub.publish(id_msg)
        
        # Log info
        self.get_logger().info(f"Tracking person ID {target_track.track_id}: center offset x={norm_offset_x:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown() 