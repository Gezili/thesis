import numpy as np
import os
#import src.data_augmentation as data_augmentation
import matplotlib.pyplot as plt
import open3d as o3d

from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from scipy.spatial.transform import Rotation

'''
def transform_pointcloud(pointcloud):

    pointcloud = pointcloud.reshape(-1,3)
    rot = Rotation.from_euler('xyz', [0, .7854, 4.7124], degrees=False)

    pointcloud = np.matmul(np.linalg.inv(rot.as_matrix()), pointcloud.T).T + np.array([0.75, -0.75, -0.75])

    rot_mat = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    pointcloud = np.matmul(rot_mat, pointcloud.T).T
    pointcloud = pointcloud.reshape(480, 640, 3)

    return pointcloud
'''

def process_pc(input_dir, masks_filepath, output_dir, plot=False):

    images = os.listdir(input_dir)
    N = len(images)

    rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)

    for i, img_file in enumerate(images):
        pc = np.load(f'{input_dir}/{img_file}', allow_pickle=True, encoding='bytes').item()

        # RGB
        rgb_img = pc['rgb']
        rgb_imgs[i] = rgb_img#data_augmentation.standardize_image(rgb_img)

        # XYZ
        xyz_imgs[i] = pc['xyz_transformed']



    with open(masks_filepath, 'rb') as f:
        all_seg_masks = np.load(f, allow_pickle=True)

    for image in range(all_seg_masks.shape[0]):


        for mask in np.unique(all_seg_masks[image,...])[1:]:



            x, y = np.where(all_seg_masks[image,...] == mask)
            xyz = np.squeeze(xyz_imgs[image,...])
            xyz = xyz[x,y]
            rgb = rgb_imgs[image,...][x,y]

            xyz_imgs[xyz_imgs == 0] = np.nan

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
                    [616.36529541,   0.        , 310.25881958],
                    [  0.        , 616.20294189, 236.59980774],
                    [  0.        ,   0.        ,   1.        ]
                ])
            }

            with open(f'{output_dir}/{image}_{int(mask)}.npy', 'wb') as f:
                np.save(f, save)

    if plot:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb/255)

        o3d.visualization.draw_geometries([pcd])
        plt.show()

if __name__ == '__main__':

    process_pc(
        input_dir='./data/pc',
        masks_filepath='./data/segmentation_masks/mask.npy',
        output_dir='./pytorch_6dof-graspnet/demo/data',
        plot=True
    )