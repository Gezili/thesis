import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group_name = "hand"
group = moveit_commander.MoveGroupCommander(group_name)

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

joint_goal = group.get_current_joint_values()
joint_goal[0] = 0.03
joint_goal[1] = 0.03
group.go(joint_goal, wait=True)
group.stop()