# Base image
FROM ubuntu:24.04

# Set locale and non-interactive frontend
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install core dependencies, ROS2, GStreamer, and Python tools in one go
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      # Localization & ROS prerequisites
      locales curl gnupg lsb-release git \
      # Python & build tools
      python3-pip build-essential software-properties-common \
      # GStreamer core + plugins + tools
      python3-gi gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0 \
      gstreamer1.0-tools gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly gstreamer1.0-libav  gstreamer1.0-x && \
    # Generate locale
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 && \
    # Add ROS2 Jazzy repo key and list
    mkdir -p /etc/apt/keyrings && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /etc/apt/keyrings/ros2-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/ros2-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    # Install ROS2 desktop and Python libs
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-jazzy-desktop python3-colcon-common-extensions \
      ros-jazzy-cv-bridge && \
    pip3 install --upgrade --break-system-packages \
      colcon-common-extensions \
      ultralytics==8.0.134 \
      deep_sort_realtime \
      opencv-python==4.7.0.72 \
      torch torchvision \
      timm && \
    # Cleanup apt caches
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN mkdir -p /home/robot/models && \
    echo "INFO: Cloning MiDaS repository into /home/robot/models/MiDaS..." && \
    git clone https://github.com/isl-org/MiDaS.git /home/robot/models/MiDaS && \
    echo "INFO: MiDaS repository cloned."

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '# export ROS_DOMAIN_ID=<RELBot_ID>  # set per robot' >> /root/.bashrc

# Default to Bash
CMD ["/bin/bash"]
