import rospy
import moveit_commander
import geometry_msgs.msg
import sys

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

scene = moveit_commander.PlanningSceneInterface()

box_pose = geometry_msgs.msg.PoseStamped()
box_pose.position.x = 0
box_pose.position.y = 0
box_pose
box_pose.pose.orientation.w = 1.0
box_pose.pose.position.z = 0.11 # above the panda_hand frame
box_name = "box"
scene.add_box(box_name, box_pose, size=(7.5, 7.5, 3.5))