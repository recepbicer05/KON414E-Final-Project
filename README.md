# KON414E-Final-Project
This projected is developed to apply FastSlam Algorithm in ROS. There are several different codes which included to project.

## ground_truth.xml
This XML snippet adds a Gazebo plugin to a robot's URDF or XACRO file for ground truth data. It publishes the robot's state (position, velocity, etc.) to the /ground_truth/state topic at a rate of 30 Hz, with no Gaussian noise applied. The plugin uses the libgazebo_ros_p3d.so file and operates under the ground_truth namespace.

## IMU Based Odometry
The IMU based odomotery requested in the project was provided with the robot_localization package. The location of the package is shown in the diagram. Thanks to this package, odometry is provided with IMU. Plots of IMU data were also shown in the 2nd assignment.

## aruco_detector.py
This script is used to detect Aruco markers in real-time, estimate their poses, transform them into the desired coordinate frame (from camera to odometry), and publish the transformed poses as a PoseArray message. This can be used for various robotics applications requiring marker-based localization or tracking.

## aruco_random_spawner.py
This Python script is a ROS (Robot Operating System) node designed to spawn random Aruco markers in a Gazebo simulation environment. The markers are spawned at random positions within a specified range, and each marker's position and orientation are determined dynamically. The goal is to create a random distribution of Aruco markers in the Gazebo simulation environment for use in tasks like visual localization, mapping, or navigation.

## accuracy.py
This ROS node continuously computes the Euclidean localization error by comparing the true and estimated positions of the robot. It publishes the error to a ROS topic for further analysis or monitoring, allowing users to track the performance of the robot’s localization system over time.

## fastslam.py
The code enables a robot to simultaneously localize itself and map its environment by using a set of particles to represent possible states and landmarks, refining its position and map through observations and motion updates.

