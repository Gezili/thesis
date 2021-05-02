import sys
import rospy

from get_kinect_data import RGBImage

def main(args):

  rospy.init_node('rgb', anonymous=True)

  imgs = RGBImage()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)