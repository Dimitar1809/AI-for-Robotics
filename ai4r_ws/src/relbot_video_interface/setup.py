from setuptools import setup
import os
from glob import glob

package_name = 'relbot_video_interface'

# Function to get all files in a directory recursively
def get_data_files():
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/video_interface.launch.py']),
    ]
    
    # Add weights directory and all its contents
    weights_dir = 'weights'
    for root, dirs, files in os.walk(weights_dir):
        for file in files:
            file_path = os.path.join(root, file)
            install_path = os.path.join('share', package_name, root)
            data_files.append((install_path, [file_path]))
    
    return data_files

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=get_data_files(),
    install_requires = ['setuptools'],
    zip_safe=True,
    maintainer='Bavantha Udugama',
    maintainer_email='b.udugama@utwente.nl',
    description='Capture GStreamer video and publish object positions to RELBot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_interface = relbot_video_interface.video_interface_node:main',
            'person_tracker = relbot_video_interface.person_tracker:main',
            'person_follow = relbot_video_interface.person_follow:main'
        ],
    },
)
