# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='relbot_video_interface',
#             executable='video_interface',
#             name='video_interface',
#             output='screen',
#             parameters=[
#                 {'gst_pipeline': (
#                     'udpsrc port=5000 caps="application/x-rtp,media=video,'
#                     'encoding-name=H264,payload=96" ! '
#                     'rtph264depay ! avdec_h264 ! videoconvert ! '
#                     'video/x-raw,format=RGB ! appsink name=sink'
#                 )}
#             ],
#         ),
#     ])
# In your video_interface.launch.py
from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    # Path to your video file INSIDE the Docker container (set up by volume mount)
    video_file_path_in_container = "/opt/mounted_videos/output_2.mp4" 

    # GStreamer pipeline to read from the file
    gst_pipeline_for_file = (
        f"filesrc location={video_file_path_in_container} ! decodebin ! "
        "videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
    )

    video_interface_node = Node(
        package='relbot_video_interface',
        executable='video_interface', # Script/entry point name
        name='video_interface_from_file', 
        parameters=[{
            'gst_pipeline': gst_pipeline_for_file, # <<<< KEY CHANGE HERE
            # Ensure other parameters are correct
            'yolo_weights': 'best.pt', 
            'MiDaS_weights': 'dpt_swin2_tiny_256.pt',
            'conf_threshold': 0.5, 
        }]
    )
    return LaunchDescription([video_interface_node])