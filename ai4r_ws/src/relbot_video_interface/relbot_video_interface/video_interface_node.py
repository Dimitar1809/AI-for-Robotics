# #!/usr/bin/env python3

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# MiDaS imports
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        self._declare_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._init_models()
        self._init_gstreamer()
        self._init_video_writer()

        # Timer at 30Hz
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized and streaming at 30Hz')

    def _declare_parameters(self):
        self.declare_parameter('gst_pipeline', (
            'udpsrc port=5000 caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96" ! '
            'rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw,format=RGB ! appsink name=sink'
        ))
        self.declare_parameter('yolo_weights', 'best.pt')
        self.declare_parameter('MiDaS_weights', 'dpt_hybrid_384.pt')
        self.declare_parameter('output_video', 'output.mp4')

    def _init_models(self):
        # YOLOv8 detection model
        pkg_dir = Path(get_package_share_directory('relbot_video_interface'))
        yolo_path = pkg_dir / 'weights' / self.get_parameter('yolo_weights').value
        self.model = YOLO(str(yolo_path))
        self.model.fuse()

        # MiDaS depth model
        midas_path = pkg_dir / 'weights' / self.get_parameter('MiDaS_weights').value
        self.model_midas = DPTDepthModel(path=str(midas_path), non_negative=True)
        self.model_midas.to(self.device).eval()

        # Transformation for DPT_Hybrid
        self.midas_transform = Compose([
            Resize(384, 384, keep_aspect_ratio=True, ensure_multiple_of=32),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ])

        # DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=3)

    def _init_gstreamer(self):
        Gst.init(None)
        pipeline_str = self.get_parameter('gst_pipeline').value
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        # Drop late frames (keep only latest)
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.bridge = cv2  # placeholder if CvBridge needed elsewhere

    def _init_video_writer(self):
        output_name = self.get_parameter('output_video').value
        self.video_path = Path(os.getcwd()) / output_name
        self.video_writer = None
        self.frame_count = 0
        self.get_logger().info(f"Recording output video to: {self.video_path}")

    def on_timer(self):
        sample = self.sink.emit('pull-sample')
        if not sample:
            return

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value('width')
        height = caps.get_value('height')

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return

        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        buf.unmap(mapinfo)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write to output video
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.video_path), fourcc, 30.0, (width, height)
            )
        self.video_writer.write(frame)

        # Detection
        results = self.model.predict(rgb, conf=0.5, iou=0.5, classes=[1])
        detections = results[0]

        # Depth estimation
        input_tensor = self.midas_transform(frame).to(self.device)
        with torch.no_grad():
            depth_pred = self.model_midas(input_tensor)
            depth_map = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1), size=(height, width),
                mode='bicubic', align_corners=False
            ).squeeze().cpu().numpy()

        # Process detections and track
        det_list = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id != 1 or conf < 0.5:
                continue
            w, h = x2 - x1, y2 - y1
            det_list.append([[x1, y1, w, h], conf, cls_id])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Tracking could be added here: self.tracker.update_tracks(...)

        # Compute center and relative depth
        if det_list:
            x1, y1, w, h = det_list[0][0]
            cx, cy = x1 + w / 2, y1 + h / 2
            depth_value = float(depth_map[int(cy), int(cx)])

            msg = Point(x=cx, y=0.0, z=depth_value)
            self.position_pub.publish(msg)
            self.get_logger().debug(f"Published position: {msg}")

        # Optional debug display
        # cv2.imshow('Output', frame)
        # cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.set_state(Gst.State.NULL)
        if self.video_writer:
            self.video_writer.release()
            self.get_logger().info('Output video closed.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()