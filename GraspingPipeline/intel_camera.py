import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import cv2

from mpl_toolkits.mplot3d import Axes3D

MAX_DEPTH = 1200

def process_intel_data(filepath, plot=False):

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(filepath)
    config.enable_all_streams()

    profile = pipeline.start(config)

    frame = pipeline.wait_for_frames()

    align = rs.align(rs.stream.color)
    frame = align.process(frame)

    depth = frame.get_depth_frame()
    rgb = frame.get_color_frame()

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    intrinsics = rgb.profile.as_video_stream_profile().intrinsics

    pc = rs.pointcloud()

    pc.map_to(rgb)
    points = pc.calculate(depth)

    rgb = np.asanyarray(rgb.get_data())
    depth = np.asanyarray(depth.get_data())

    pc_xyz = np.asanyarray(points.get_vertices())
    pc_xyz = np.stack([pc_xyz['f0'], pc_xyz['f1'], pc_xyz['f2']]).T

    rgb[depth > MAX_DEPTH] = 0

    pc_xyz = pc_xyz.reshape(rgb.shape)
    pc_xyz[depth > MAX_DEPTH] = 0

    rgb = rgb[:, 160:-160, :]
    pc_xyz = pc_xyz[:, 160:-160]

    rgb = cv2.resize(rgb, (640, 480))

    #Not sure how to downsample point cloud nicely - so here we use nearest neighbor. Tried bilinear and bicubic with horrible results. Other downsampling approaches are not nice and adaptive the way that we want unfortunately
    pc_xyz = cv2.resize(pc_xyz, (640, 480), interpolation = cv2.INTER_NEAREST)

    savefile = {}
    savefile['rgb'] = rgb
    savefile['xyz'] = pc_xyz

    if plot:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ahh = pc_xyz.reshape(-1,3)
        rgb = rgb.reshape(-1,3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ahh)
        pcd.colors = o3d.utility.Vector3dVector(rgb/255)
        o3d.visualization.draw_geometries([pcd])

    return savefile

if __name__ == '__main__':

    process_intel_data("./data/sensor_data/20201107_115836.bag")