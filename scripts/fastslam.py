#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseArray, Pose
from tf.transformations import quaternion_from_euler

class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.landmarks = []  # List of detected landmarks for this particle

    def predict(self, odom):
        """
        Predict step: Update particle state based on odometry
        """
        dx = odom.twist.twist.linear.x
        dtheta = odom.twist.twist.angular.z

        # Add noise to motion model for realism
        noise_factor = 0.1
        dx_noise = np.random.normal(0, noise_factor)
        dtheta_noise = np.random.normal(0, noise_factor)

        self.x += (dx + dx_noise) * np.cos(self.theta)
        self.y += (dx + dx_noise) * np.sin(self.theta)
        self.theta += dtheta + dtheta_noise

    def update_landmarks(self, observed_landmarks):
        """
        Update landmarks using observed data
        """
        for obs in observed_landmarks:
            obs_x = obs.position.x
            obs_y = obs.position.y

            # If no landmarks exist, initialize
            if len(self.landmarks) < len(observed_landmarks):
                self.landmarks.append([obs_x, obs_y, 0.5])  # Initialize with mean_x, mean_y, variance

            # Update landmark position using observed data (Kalman-like update)
            for i in range(len(self.landmarks)):
                mean_x, mean_y, variance = self.landmarks[i]
                dx = obs_x - mean_x
                dy = obs_y - mean_y

                # Update the mean and variance
                self.landmarks[i][0] += 0.1 * dx  # Update mean_x
                self.landmarks[i][1] += 0.1 * dy  # Update mean_y
                self.landmarks[i][2] = max(0.1, variance * 0.9)  # Update variance

class FastSLAM:
    def __init__(self):
        rospy.init_node("fastslam", anonymous=True)

        # Parameters
        self.num_particles = 1000
        self.map_size = 1000  # Map size in grid cells
        self.resolution = 0.1  # Map resolution (meters per cell)

        # Initialize particles
        self.particles = [
            Particle(
                x=np.random.uniform(-5, 5),
                y=np.random.uniform(-5, 5),
                theta=np.random.uniform(0, 2 * np.pi),
                weight=1.0 / self.num_particles,
            )
            for _ in range(self.num_particles)
        ]

        # Initialize global landmark list to track landmarks across all particles
        self.global_landmarks = []  # List of all detected landmarks across particles

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.landmark_sub = rospy.Subscriber("/aruco_marker_poses", PoseArray, self.landmark_callback)

        # Publishers
        self.map_pub = rospy.Publisher("/map", OccupancyGrid, queue_size=10)
        self.particles_pub = rospy.Publisher("/particles", PoseArray, queue_size=10)

        rospy.loginfo("FastSLAM initialized")

        # Variable to hold last odometry message
        self.last_odom_msg = None

    def odom_callback(self, msg):
        """
        Callback for odometry updates
        """
        self.last_odom_msg = msg
        for particle in self.particles:
            particle.predict(msg)

    def landmark_callback(self, msg):
        """
        Callback for landmark observations
        """
        if len(msg.poses) == 0:
            rospy.loginfo("No landmarks detected.")
            return

        observed_landmarks = msg.poses

        # Add newly observed landmarks to global list, avoiding duplicates
        for obs in observed_landmarks:
            is_new = True
            for global_lm in self.global_landmarks:
                if np.abs(global_lm[0] - obs.position.x) < 0.1 and np.abs(global_lm[1] - obs.position.y) < 0.1:
                    is_new = False
                    break
                
            if is_new:
                self.global_landmarks.append([obs.position.x, obs.position.y])

        # Update particle landmarks
        for particle in self.particles:
            particle.update_landmarks(observed_landmarks)

        self.resample_particles()
        self.publish_particles()
        self.publish_map()

    def resample_particles(self):
        """
        Resample particles based on their weights
        """
        weights = np.array([p.weight for p in self.particles])
        weights /= np.sum(weights)

        if np.any(np.isnan(weights)):
            rospy.logwarn("NaN detected in weights. Resetting.")
            weights.fill(1.0 / self.num_particles)

        indices = np.random.choice(len(self.particles), size=self.num_particles, p=weights)
        self.particles = [self.particles[i] for i in indices]

    def publish_particles(self):
        """
        Publish particles as PoseArray for visualization
        """
        particle_poses = PoseArray()
        particle_poses.header.frame_id = "odom"
        particle_poses.header.stamp = rospy.Time.now()

        for p in self.particles:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            quaternion = quaternion_from_euler(0, 0, p.theta)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            particle_poses.poses.append(pose)

        self.particles_pub.publish(particle_poses)

    def publish_map(self):
        """
        Publish a simple occupancy grid map with the detected landmarks
        """
        grid = OccupancyGrid()
        grid.header.frame_id = "odom"
        grid.header.stamp = rospy.Time.now()
        grid.info.resolution = self.resolution
        grid.info.width = self.map_size
        grid.info.height = self.map_size
        grid.info.origin.position.x = -self.map_size * self.resolution / 2
        grid.info.origin.position.y = -self.map_size * self.resolution / 2

        # Initialize with -1 (unknown cells)
        data = [-1] * (self.map_size * self.map_size)

        # Update grid cells based on global landmarks
        for lm in self.global_landmarks:
            x, y = lm

            # Convert to map grid coordinates
            map_x = int((x + self.map_size * self.resolution / 2) / self.resolution)
            map_y = int((y + self.map_size * self.resolution / 2) / self.resolution)

            # Ensure we are within bounds
            if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                index = map_y * self.map_size + map_x
                data[index] = 100  # Mark as occupied cell

        grid.data = data
        self.map_pub.publish(grid)


if __name__ == "__main__":
    try:
        slam = FastSLAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
