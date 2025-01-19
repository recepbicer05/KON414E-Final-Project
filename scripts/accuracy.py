#!/usr/bin/env python3
# Recep Bicer

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np

class EvaluateAccuracyNode:
    def __init__(self):
        rospy.init_node('evaluate_accuracy_node', anonymous=True)
        
        # Initial States
        self.true_pose = None
        self.estimated_pose = None
        
        # Sub-Pub
        rospy.Subscriber("/ground_truth/state", Odometry, self.true_pose_callback)
        rospy.Subscriber("/odom", Odometry, self.estimated_pose_callback)
        rospy.Timer(rospy.Duration(1), self.calculate_accuracy)
        self.accuracy_pub = rospy.Publisher("/accuracy", Float64, queue_size=10)

    def true_pose_callback(self, msg):
        self.true_pose = msg.pose.pose

    def estimated_pose_callback(self, msg):
        self.estimated_pose = msg.pose.pose

    def calculate_accuracy(self, event):
        if self.true_pose is None or self.estimated_pose is None:
            rospy.logwarn("True pose or estimated pose not received yet.")
            return

        true_x = self.true_pose.position.x
        true_y = self.true_pose.position.y
        estimated_x = self.estimated_pose.position.x
        estimated_y = self.estimated_pose.position.y

        # Euclidian
        error = np.sqrt((true_x - estimated_x)**2 + (true_y - estimated_y)**2)
        self.accuracy_pub.publish(error)
        rospy.loginfo(f"Localization Error: {error:.6f} meters")

if __name__ == "__main__":
    try:
        EvaluateAccuracyNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
