#!/usr/bin/env python
# TEST SCRIPT TO CHECK AND VERIFY 
# GAZEBO GROUND TRUTH POSE for FINAL_IRIS model 

import rospy
from gazebo_msgs.msg import ModelStates

class GT_SUB:
    def __init__(self):
        rospy.init_node('iris_ground_truth_pose', anonymous=True)

        self.target = "final_iris"   # model name to track
        
        self.sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.callback
        )

    def callback(self, msg: ModelStates):
        if self.target not in msg.name:
            # model not spawned yet or name mismatch
            rospy.logwarn_throttle(2.0, f"{self.target} not found in model_states")
            return
        
        idx = msg.name.index(self.target)
        pose = msg.pose[idx]

        # log ground truth pose
        rospy.loginfo(f"{self.target} pose: {pose}")
        

        # OR
        # for target in msg.name:
        #     i = msg.name.index(target)
        #     rospy.loginfo(f"pose {i} : {msg.pose[i]}")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        GT_SUB().run()
    except rospy.ROSInterruptException:
        pass
