#!/usr/bin/env python3
# Creates transform from the pose of the gazebo GT to display in rviz as a frame
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros

class GazeboTF:
    def __init__(self):
        self.parent_frame = rospy.get_param("~parent_frame", "world")
        self.child_frame  = rospy.get_param("~child_frame",  "gazebo_gt")
        self.br = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/iris_gt_pose", PoseStamped, self.cb, queue_size=1)

    def cb(self, ps: PoseStamped):
        t = TransformStamped()
        t.header.stamp = ps.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id  = self.child_frame
        t.transform.translation.x = ps.pose.position.x
        t.transform.translation.y = ps.pose.position.y
        t.transform.translation.z = ps.pose.position.z
        t.transform.rotation      = ps.pose.orientation
        self.br.sendTransform(t)

if __name__ == "__main__":
    rospy.init_node("gazebo_pose_to_tf")
    GazeboTF()
    rospy.spin()
