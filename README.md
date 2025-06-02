## To make the RelBot to follow the person with helmet run the same commands as before (given in the github: https://github.com/UAV-Centre-ITC/AI4R_RELBot ),

One small change is to use:
cd /ai4r_ws

colcon build --symlink-install <-- this is different 

source install/setup.bash

ros2 launch relbot_video_interface video_interface.launch.py

It should be noted that in order to use the different ML models, they have to be in the weights folder, next to the **relbot_video_interface folder.

Also, copying the midas folder is needed:

docker cp ~/AI4R_RELBot/MiDaS_v3.1_contents/. 

relbot_ai4r_assignment1:/home/robot/models/MiDaS/**
