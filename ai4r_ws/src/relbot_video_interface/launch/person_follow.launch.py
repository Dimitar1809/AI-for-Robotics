from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('relbot_video_interface')
    weights_path = os.path.join(pkg_dir, 'weights', 'yolov8n.pt')
    
    return LaunchDescription([
        Node(
            package='relbot_video_interface',
            executable='person_follow',
            name='person_follow',
            output='screen',
            parameters=[{
                'yolo_weights': weights_path,
                'midas_model_type': 'MiDaS_small',
                'desired_distance': 2.0,
                'camera_topic': '/camera/image_raw',
                'conf_threshold': 0.5
            }]
        )
    ]) 