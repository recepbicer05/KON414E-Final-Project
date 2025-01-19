#!/usr/bin/env python3
# Recep Bicer

import rospy
import random
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import xml.etree.ElementTree as ET

def spawn_random_aruco_markers():
    rospy.init_node('random_aruco_marker_spawner')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')

    try:
        # This service will be provide to spawn .sdf format files
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Path of the Aruco Marker 
        model_path = '/home/recep/catkin_ws/src/aruco_navigation/models/aruco_marker1/model.sdf'
        with open(model_path, 'r') as file:
            model_xml = file.read()

        # Determine the number of Markers randomly
        num_markers = random.randint(40, 70)

        # Spawn the markers to random positions
        for i in range(num_markers):
            model_name = f'aruco_marker_{i+1}'

    
            pose = Pose()
            while True:
                pose.position.x = random.uniform(-15.0, 250.0)
                pose.position.y = random.uniform(-4.0, 4.0)
                # This line will prevent the collison between random aruco's with robot
                if not (-0.2 <= pose.position.x <= 0.2 and -0.2 <= pose.position.y <= 0.2):
                        break
            pose.position.z = 0.5  
            pose.orientation.w = random.uniform(0.0,1.0)
            spawn_model(model_name, model_xml, "", pose, "world")

            rospy.loginfo(f"Model '{model_name}' successfully spawned at position: ({pose.position.x}, {pose.position.y}, {pose.position.z})")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    spawn_random_aruco_markers()
