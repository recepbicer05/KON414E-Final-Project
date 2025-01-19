#!/usr/bin/env python3
# Recep Bicer

import cv2
import cv2.aruco as aruco
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs

class ArucoDetector:
    def __init__(self):
        # ROS Node Initialization
        rospy.init_node("aruco_detector", anonymous=True)

        # Subscribers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        # Publishers
        self.pose_pub = rospy.Publisher("/aruco_marker_poses", PoseArray, queue_size=10)

        # Other Initializations
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_size = 0.5  # Updated Aruco marker size to 0.5 meters
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Wait for Camera Info
        rospy.loginfo("Waiting for camera_info to be received...")
        while self.camera_matrix is None or self.dist_coeffs is None:
            rospy.sleep(0.1)
        rospy.loginfo("Camera info received. Ready to process images.")

    def camera_info_callback(self, msg):
        # Callback to handle CameraInfo messages
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def image_callback(self, msg):
        # Callback to handle Image messages
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn("Waiting for camera_info...")
            return

        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Convert to Grayscale for Aruco Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define Aruco Dictionary and Parameters {This is the most important part, dictionary has to be defined correctly!!}
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters_create()

        # Detect Aruco Markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # Draw detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            # Prepare PoseArray message
            pose_array = PoseArray()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = "depth_link"

            for i in range(len(ids)):
                rvec, tvec = rvecs[i][0], tvecs[i][0]

                # Convert rotation vector to quaternion
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)

                # Create Pose message in camera frame
                pose = Pose()
                pose.position.x = tvec[0]
                pose.position.y = tvec[1]
                pose.position.z = tvec[2]
                pose.orientation.x = quaternion[0]
                pose.orientation.y = quaternion[1]
                pose.orientation.z = quaternion[2]
                pose.orientation.w = quaternion[3]

                # Transform pose to odom frame
                transformed_pose = self.transform_pose(pose, "odom", "depth_link")
                if transformed_pose:
                    pose_array.poses.append(transformed_pose)

                rospy.loginfo(f"Marker ID: {ids[i][0]}, Position: {tvec}")

            # Publish PoseArray
            self.pose_pub.publish(pose_array)

        # Display the Image with Detections
        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    def transform_pose(self, pose, from_frame, to_frame):
        # Transform pose from one frame to another
        try:
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
            transformed_pose = tf2_geometry_msgs.do_transform_pose(PoseStamped(header=Header(frame_id=from_frame), pose=pose), transform)
            return transformed_pose.pose
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform error: {e}")
            return None

    @staticmethod
    def rotation_matrix_to_quaternion(rotation_matrix):
        # Convert a rotation matrix to quaternion
        q = np.zeros(4)
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            q[1] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            q[2] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                q[3] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                q[2] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                q[3] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                q[0] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                q[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                q[0] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                q[1] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                q[2] = 0.25 * s
        return q

if __name__ == "__main__":
    try:
        detector = ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
