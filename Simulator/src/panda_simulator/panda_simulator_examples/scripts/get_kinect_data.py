#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
import tf2_ros as tf2

from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError

from scipy.spatial.transform import Rotation

def transform_pointcloud(pointcloud):

    rot = Rotation.from_euler('xyz', [0, .7854, 4.7124], degrees=False)

    pointcloud = np.matmul(np.linalg.inv(rot.as_dcm()), pointcloud.T).T + np.array([0.75, -0.75, -0.75])

    rot_mat = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    pointcloud = np.matmul(rot_mat, pointcloud.T).T

    return pointcloud

class Points:

  def __init__(self):

    #rospy.init_node('point_transforms')

    self.bridge = CvBridge()
    self.points_sub = rospy.Subscriber("/camera/depth/points",PointCloud2,self.callback)
    self.count = 0

  def callback(self,data):
    points = []
    pc = pc2.read_points(data, field_names = ("x", "y", "z"))

    for point in pc:
      points.append(point)

    #points = transform_pointcloud(points)
    print(len(points[0]))



    points = np.array(points)
    points_transformed = transform_pointcloud(points)

    print(points_transformed[0])
    print(points_transformed[-1])
    print(points_transformed[10000])
    print(points_transformed[20000])
    #test = np.array([point[0] for point in points]).reshape(480, 640)
    test = points.reshape(480, 640, 3)
    #cv2.imshow("Image window", test)
    #cv2.waitKey(3)

    if self.count % 30 == 0:

      np.save('xyz.npy', test)
      np.save('xyz_transformed.npy', points_transformed.reshape(480, 640, 3))
    self.count += 1



class DepthImage:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/depth/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      data.encoding = "mono16"
      cv_image = self.bridge.imgmsg_to_cv2(data, 'mono8')
    except CvBridgeError as e:
      print(e)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)


class RGBImage:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
    self.is_saved = False
    self.count = 0

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
    except CvBridgeError as e:
      print(e)

    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)

    if self.count % 30 == 0:
      cv2.imwrite('rgb.png', cv_image)
    self.count += 1

def main(args):

  rospy.init_node('points', anonymous=True)

  #f = RGBImage()
  #rospy.init_node('numpy', anonymous=True)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)