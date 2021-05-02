import rospy
import tf2_ros
import geometry_msgs.msg

if __name__ == '__main__':

    rospy.init_node('yikes')
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)

    #rospy.wait_for_service('spawn')

    trans_grip_center_p8 = buffer.lookup_transform("panda_link2", "panda_link0", rospy.Time(0))