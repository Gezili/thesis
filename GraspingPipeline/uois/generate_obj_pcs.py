import numpy as np
import os
import src.data_augmentation as data_augmentation
import matplotlib.pyplot as plt
import open3d as o3d

from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial

example_images_dir = os.listdir('./data')
N = len(example_images_dir)

rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)

def denoise_pointcloud(points, rgb=None):

    rejection_thresh = .8
    num_points_to_keep = int(points.shape[0]*rejection_thresh)
    delta = 0.2

    tree = spatial.cKDTree(points)
    local_densities = []

    for point in points:

        indices = tree.query_ball_point(point, delta)
        local_density = np.linalg.norm(points[indices] - point)

        local_densities.append(local_density)

    indices_to_keep = np.argsort(local_densities)[:num_points_to_keep]

    if not rgb:
        return points[indices_to_keep]

    return points[indices_to_keep], rgb[indices_to_keep]


for i, img_file in enumerate(example_images_dir):
    d = np.load(f'./data/{img_file}', allow_pickle=True, encoding='bytes').item()

    # RGB
    rgb_img = d['rgb']
    rgb_imgs[i] = rgb_img#data_augmentation.standardize_image(rgb_img)

    # XYZ
    xyz_imgs[i] = d['xyz']

with open('./mask.npy', 'rb') as f:
    all_seg_masks = np.load(f, allow_pickle=True)

#for i in ran

for image in range(1):#all_seg_masks.shape[0]):

    for mask in [2]:#np.unique(all_seg_masks[image,...])[:1]:

        print(mask)

        #x, y = np.where(all_seg_masks[image,...] == mask)
        x, y = np.where(all_seg_masks[image,...] == 2)

        xyz = np.squeeze(xyz_imgs[image,...])
        xyz = xyz[x,y]
        rgb = rgb_imgs[image,...][x,y]

        xyz_imgs[xyz_imgs == 0] = np.nan
        #xyz = denoise_pointcloud(xyz)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb/255)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)

        xyz = np.asarray(pcd.points)

        save = {
            'base_to_camera_rt': 'base_to_camera_rt',
            'smoothed_object_pc': xyz,
            'depth': xyz_imgs[image,...,2],
            'image': rgb_imgs[image,...].astype('uint8'),
            #TODO change intrinsics matrix
            'intrinsics_matrix': np.array([
                [920.944,   0, 646.671],
                [0, 920.635, 358.757],
                [  0,   0,   1]
            ])
        }

        with open(f'./graspdata/{image}_{int(mask)}.npy', 'wb') as f:
            np.save(f, save)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb/255)




o3d.visualization.draw_geometries([pcd])



plt.show()