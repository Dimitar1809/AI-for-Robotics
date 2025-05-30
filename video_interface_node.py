#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2
import ultralytics  # Assuming ultralytics is installed for YOLOv8
from ultralytics import YOLO
import torch
from cv_bridge import CvBridge

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

        # Get parameter values
        yolo_weights = self.get_parameter('yolo_weights').get_parameter_value().string_value

        # Get package share directory for weights
        pkg_dir = get_package_share_directory('relbot_video_interface')
        weights_path = os.path.join(pkg_dir, 'weights', yolo_weights)

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
        self.model = YOLO(weights_path)
        self.model.fuse()  # fuse model layers for speed
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=3)

        # Timer: fires at ~30Hz to pull frames and publish positions
        # The period (1/30) sets how often on_timer() is called
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

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
        buf.unmap(mapinfo)

        # Display the raw input frame for debugging
        cv2.imshow('Input Stream', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


        # Load a pretrained model on COCO datatset
        results = self.model.predict(frame, conf=0.5, iou=0.5, classes=[1])  # class 0 = person
        detections = results[0]
        # Prepare detections in DeepSORT format: [[xmin,ymin,width,height], confidence, class_id]
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
            else:
                print("no helmet detected")

        # TODO: Insert detection/tracking logic here to compute object position
        # For demonstration, here we are publishing a dummy Point at origin
                # Compute and publish object position:
        # x = horizontal center coordinate of the object
        # y = unused (flat-ground assumption)
        # z = object area (controller caps at 10000 to stop robot when object is too large)
        msg = Point()
        msg.x = 200.0  # object center x-coordinate
        msg.y = 0.0  # y-coordinate unused
        msg.z = 10001.0  # object area; >10000 indicates 'too close'
        self.position_pub.publish(msg)
        # To adjust robot behavior, apply a scaling factor to 'z' (e.g., couple with depth estimation)
        # Log at debug level if needed:
        # self.get_logger().debug(f'Published position: ({msg.x}, {msg.y}, {msg.z})')

    def destroy_node(self):
        # Cleanup GStreamer resources on shutdown
        self.pipeline.set_state(Gst.State.NULL)
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