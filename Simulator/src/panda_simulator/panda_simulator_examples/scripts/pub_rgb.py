import sys
import rospy

from get_kinect_data import Points

def main(args):

  rospy.init_node('points', anonymous=True)

  imgs = Points()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)