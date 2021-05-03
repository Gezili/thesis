import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import trimesh
import itertools
import collections
import trimesh.transformations as tra
import time
import cv2

from renderer import ObjectRenderer
from mpl_toolkits.mplot3d import Axes3D

#np.random.seed(0)

def render_scene_object_renderer():

    path = './processed'
    table_path = './surfacemesh/CoffeeTable.obj'

    object_names = os.listdir(path)[12:13]
    object_paths = [f'{path}/{obj}/model.obj' for obj in object_names]

    meshes = []

    surface_mesh = trimesh.load(table_path)

    z_coordinates = surface_mesh.vertices[:,2]
    table_height = z_coordinates.max() - z_coordinates.min()

    object_paths.append(table_path)

    scales = [1]*len(object_paths)
    scales[-1] = 0.03

    renderer = ObjectRenderer(
        object_paths=list(np.array(object_paths)),
        object_scales=scales
    )

    for object in object_paths:

        meshes.append(trimesh.load(object))

    while True:

        all_poses = []

        for i, object in enumerate(object_paths[:-1]):

            mesh = meshes[i]

            homogenous_matrix = np.eye(4)

            theta = np.random.rand()*2*np.pi
            z_euler_rotation_matrix = tra.euler_matrix(0, 0, theta)

            jitter = 1
            translation = np.random.rand(2)*jitter - jitter/2
            homogenous_matrix[:2,3] = translation
            homogenous_matrix[:3,:3] = z_euler_rotation_matrix[:3,:3]

            lowest_point_in_mesh = mesh.vertices[:,2].min()

            homogenous_matrix[2,3] = -lowest_point_in_mesh
            all_poses.append(copy.deepcopy(homogenous_matrix))

        collusion_manager = trimesh.collision.CollisionManager()

        for i, object in enumerate(object_paths[:-1]):

            mesh = meshes[i]
            collusion_manager.add_object(
                name=object.split('/')[2],
                mesh=mesh,
                transform=all_poses[i]
            )

        collusions = collusion_manager.in_collision_internal(return_names=True)[1]

        objects_to_remove = []

        while collusions:

            collided_objects = list(itertools.chain.from_iterable(collusions))
            collusions_per_object = collections.Counter(collided_objects)

            most_collided_object = collusions_per_object.most_common(1)[0][0]

            objects_to_remove.append(most_collided_object)
            collusions = [c for c in collusions if most_collided_object not in c]

        object_mask = [name not in objects_to_remove for name in object_names]
        object_mask.append(True)

        homogenous_matrix = np.eye(4)
        homogenous_matrix[2, 3] = -table_height*0.015

        all_poses.append(homogenous_matrix)

        color, depth, xyz = renderer.render(
            object_paths=list(np.array(object_paths)[object_mask]),
            object_poses=list(np.array(all_poses)[object_mask])
        )

        camera_pose = renderer.camera_pose

        savefile = {}
        savefile['rgb'] = color
        savefile['xyz'] = xyz
        savefile['camera_pose'] = camera_pose
        savefile['object_paths'] = list(np.array(object_paths)[object_mask])
        savefile['object_poses'] = list(np.array(all_poses)[object_mask])
        fuckoff = renderer._to_pointcloud(depth, xyz=False)
        camera_pose = renderer.camera_pose

        with open(f'./data/{time.time()}.npy', 'wb') as f:
            np.save(f, savefile)

        cv2.imwrite(f'./data/{time.time()}.png', color)

    return color, xyz, depth

if __name__ == '__main__':


    pc, pose, color =  render_scene_object_renderer()

    R = pose[:3,:3]
    t = pose[:3,3]

    o = pc

    R = tra.euler_matrix(0, np.pi, 0)[:3,:3]@R
    o = (R.T@pc.T).T - t

    o -= np.array([0, 0, o[:,2].min()])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(o[:,0], o[:,1], o[:,2])
    plt.show()
    '''
    plt.imshow(color)
    plt.show()
    '''