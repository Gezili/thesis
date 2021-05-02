import os
import imageio
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def process_ros_data(plot=False):

    artifact_path = r'\\wsl$\Ubuntu-18.04\root\catkin_ws\src\panda_simulator\panda_simulator_examples\scripts'

    rgb_path = artifact_path + r'\rgb.png'
    xyz_path = artifact_path + r'\xyz.npy'
    xyz_wf_path = artifact_path + r'\xyz_transformed.npy'

    im_rgb = imageio.imread(rgb_path)
    im_xyz = np.load(xyz_path)
    img_xyz_transformed = np.load(xyz_wf_path)


    #im_xyz = np.transpose(np.array([im_xyz[...,1], im_xyz[...,0], im_xyz[...,2]]),(1,2,0))
    #im_xyz[...,1] = np.flip(im_xyz[...,1])

    savefile = {}
    savefile['rgb'] = im_rgb
    savefile['xyz'] = im_xyz
    savefile['xyz_transformed'] = img_xyz_transformed

    #if plot:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xyz = im_xyz.reshape(-1,3)
    rgb = im_rgb.reshape(-1,3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb/255)
    o3d.visualization.draw_geometries([pcd])

    return savefile

if __name__ == '__main__':

    process_ros_data()

