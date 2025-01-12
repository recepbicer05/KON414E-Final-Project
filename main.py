import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from fastslam import FastSLAM
from aruco_detector import ArUcoDetector
from sensor_processing import IMUProcessor, OdometryProcessor
from visualization import RVizVisualizer
import cv2
from cv_bridge import CvBridge

class FastSLAMNode:
    def __init__(self):
        rospy.init_node("fastslam_node")

        # Initialize FastSLAM and related components
        self.fastslam = FastSLAM(num_particles=100)
        self.aruco_detector = ArUcoDetector()
        self.imu_processor = IMUProcessor()
        self.odom_processor = OdometryProcessor()
        self.visualizer = RVizVisualizer()

        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.imu_sub = rospy.Subscriber("/imu", Imu, self.imu_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # Data containers
        self.latest_image = None
        self.latest_imu = None
        self.latest_odometry = None

    def image_callback(self, data):
        """Handle image data from the RGB-D camera."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")

    def imu_callback(self, data):
        """Handle IMU data."""
        self.latest_imu = data

    def odom_callback(self, data):
        """Handle odometry data."""
        self.latest_odometry = data

    def process(self):
        """Main loop for FastSLAM processing."""
        rate = rospy.Rate(10)  # 10 Hz loop
        while not rospy.is_shutdown():
            if self.latest_image is None or self.latest_imu is None or self.latest_odometry is None:
                continue

            # Step 1: Detect ArUco markers
            markers = self.aruco_detector.detect(self.latest_image)

            # Step 2: Process IMU and odometry data
            imu_data = self.imu_processor.process(self.latest_imu)
            odometry_data = self.odom_processor.process(self.latest_odometry)

            # Step 3: Update FastSLAM particles
            self.fastslam.update_particles(odometry_data, imu_data, markers)

            # Step 4: Resample particles
            self.fastslam.resample_particles()

            # Step 5: Update the map
            self.fastslam.update_map(markers)

            # Step 6: Visualize in RViz
            self.visualizer.visualize(self.fastslam)

            rate.sleep()

if __name__ == "__main__":
    try:
        fastslam_node = FastSLAMNode()
        fastslam_node.process()
    except rospy.ROSInterruptException:
        pass
