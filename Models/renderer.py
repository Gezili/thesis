from __future__ import print_function
from __future__ import absolute_import

import trimesh
import trimesh.transformations as tra
import pyrender
import numpy as np
import copy
import cv2
import h5py
import utils.sample as sample
import math
import sys
import argparse
import os
import utils

from mpl_toolkits.mplot3d import Axes3D


class ObjectRenderer:
    def __init__(self, fov=np.pi / 6, object_paths=[], object_scales=[]):
        """
        Args:
          fov: float,
        """
        self._fov = fov
        self.mesh = None
        self._scene = None
        self.tmesh = None
        self._init_scene()
        self._object_nodes = []
        self._object_means = []
        self._object_distances = []
        self._meshes = {}

        self.renderer = pyrender.OffscreenRenderer(480, 640)

        assert (isinstance(object_paths, list))
        assert (len(object_paths) > 0)

        for path, scale in zip(object_paths, object_scales):
            self._load_object(path, scale)

    def _init_scene(self):

        self._scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(
            yfov=self._fov, aspectRatio=1.0,
            znear=0.001)  # do not change aspect ratio

        theta, phi = 0, np.pi/4 + np.random.rand()*(np.pi/8)

        radius = 3 + (np.random.rand() - 1)
        x = radius*np.sin(phi)*np.cos(theta)
        y = radius*np.sin(phi)*np.sin(theta)
        z = radius*np.cos(phi)
        camera_pos = np.array([x, y, z])
        #camera_pos = np.array([0,0,10])
        #light_pos = np.array([0,0,10])
        camera_pose = tra.euler_matrix(0, phi, 0)
        camera_pose[:3,3] = camera_pos

        camera_pose[:3,:3] = camera_pose[:3,:3]

        self._scene.add(camera, pose=camera_pose, name='camera')

        self.camera_pose = camera_pose

        light = pyrender.SpotLight(color=np.ones(4),
                                   intensity=50.,
                                   innerConeAngle=np.pi / 16,
                                   outerConeAngle=np.pi / 6.0)
        self._scene.add(light, pose=camera_pose, name='light')

    def _load_object(self, path, scale=1.0):
        obj = sample.Object(path)

        if scale != 1.0:
            obj.rescale(scale)
            print('rescaling with scale', scale)

        tmesh = obj.mesh
        tmesh_mean = np.mean(tmesh.vertices, 0)
        tmesh.vertices -= np.expand_dims(tmesh_mean, 0)

        self.tmesh = copy.deepcopy(tmesh)
        mesh = pyrender.Mesh.from_trimesh(tmesh)

        self._meshes[path] = self._scene.add(mesh, name='object')

    def get_camera_pose(self):
        return self.camera_pose

    def to_point_cloud_world_frame(self, depth):

        pc = self._to_pointcloud(depth, xyz=False)
        R = self.camera_pose[:3,:3]
        t = t = self.camera_pose[:3,3]

        return (R.T@tra.euler_matrix(0, 0, np.pi)[:3, :3]@pc.T).T #R.T@pc + R.T@t

    def _to_pointcloud(self, depth, xyz=True):
        fy = fx = 0.5 / np.tan(self._fov * 0.5)  # aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        if xyz:
            mask = np.where(depth > -1)
        else:
            mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = -normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        test = self.camera_pose
        test[:3, :3] = test[:3, :3].T
        test[:3,3] = -test[:3,3]

        if xyz:
            return np.vstack((world_x, world_y, world_z)).T.reshape([height, width, 3])
        else:
            return np.vstack((world_x, world_y, world_z)).T
    def render(self, object_paths, object_poses, xyz=True):

        self._init_scene()

        for i, object in enumerate(object_paths):

            self._scene.add_node(self._meshes[object])
            self._scene.set_pose(self._meshes[object], object_poses[i])

        color, depth = self.renderer.render(self._scene)

        color = np.rot90(color, -1)
        depth = np.rot90(depth, -1)

        if xyz:
            xyz = self._to_pointcloud(depth)

            return color, depth, xyz

        return color, depth