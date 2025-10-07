#!/usr/bin/env python3
# this is the map overlay for the gazebo world palm_oil_estate from the SUPER FAST NAV repository by Hammock Robotics
import rospy
from visualization_msgs.msg import Marker, MarkerArray

class TreeMarkerPublisher:
    def __init__(self):
        rospy.init_node('tree_marker_publisher', anonymous=True)

        # Publisher
        self.pub = rospy.Publisher("tree_markers", MarkerArray, queue_size=1)

        # Params
        self.spacing        = rospy.get_param("~spacing", 9.0)     # meters
        self.grid_size      = rospy.get_param("~grid_size", 5)     # 5x5
        self.trunk_height   = rospy.get_param("~trunk_height", 5.0)
        self.trunk_diameter = rospy.get_param("~trunk_diameter", 0.5)
        self.frame_id       = rospy.get_param("~frame_id", "world")  # or "world"

        # Tree bases at z=0, first quadrant, skip origin by starting at 1
        self.tree_positions = [
            (i * self.spacing, j * self.spacing, 0.0)
            for i in range(0, self.grid_size)
            for j in range(0, self.grid_size)
            if not (i == 0 and j == 0)
        ]

        # Build markers once; we’ll just update timestamps before publishing
        self.marker_array = self.build_markers()

    def build_markers(self):
        ma = MarkerArray()
        for i, (x, y, z) in enumerate(self.tree_positions):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.ns = "trees"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            # Cylinder size
            m.scale.x = self.trunk_diameter
            m.scale.y = self.trunk_diameter
            m.scale.z = self.trunk_height

            # Color (opaque brown)
            m.color.r, m.color.g, m.color.b, m.color.a = (0.6, 0.3, 0.0, 1.0)

            # Pose: center the cylinder so its base sits on z=0
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z + self.trunk_height / 2.0
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0

            # Keep forever (no lifetime)
            # m.lifetime = rospy.Duration(0)

            ma.markers.append(m)
        return ma

    def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            for m in self.marker_array.markers:
                m.header.stamp = now
            self.pub.publish(self.marker_array)
            rate.sleep()

if __name__ == "__main__":
    node = TreeMarkerPublisher()
    node.run()
