import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import sys
import json
from time import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# My libraries. Ugly hack to import from sister directory
import src.data_augmentation as data_augmentation
import src.segmentation as segmentation
import src.evaluation as evaluation
import src.util.utilities as util_
import src.util.flowlib as flowlib



uois3d_config = {

    # Padding for RGB Refinement Network
    'padding_percentage' : 0.25,

    # Open/Close Morphology for IMP (Initial Mask Processing) module
    'use_open_close_morphology' : True,
    'open_close_morphology_ksize' : 9,

    # Largest Connected Component for IMP module
    'use_largest_connected_component' : True,

}

rrn_config = {

    # Sizes
    'feature_dim' : 64, # 32 would be normal
    'img_H' : 224,
    'img_W' : 224,

    # architecture parameters
    'use_coordconv' : False,

}

dsn_config = {

    # Sizes
    'feature_dim' : 64, # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters' : 10,
    'epsilon' : 0.05, # Connected Components parameter
    'sigma' : 0.02, # Gaussian bandwidth parameter
    'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
    'subsample_factor' : 5,

    # Misc
    'min_pixels_thresh' : 500,
    'tau' : 15.,

}

checkpoint_dir = 'C:\\Projects\\Thesis\\Segmentation\\uois\\models\\'

# TODO: change this to directory of downloaded models
dsn_filename = 'C:\\Projects\\Thesis\\Segmentation\\uois\\models\\DepthSeedingNetwork_3D_TOD_checkpoint.pth'
rrn_filename = 'C:\\Projects\\Thesis\\Segmentation\\uois\\models\\RRN_OID_checkpoint.pth'
uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
uois_net_3d = segmentation.UOISNet3D(uois3d_config,
                                     dsn_filename,
                                     dsn_config,
                                     rrn_filename,
                                     rrn_config
                                    )

example_images_dir = os.listdir('./data')

N = len(example_images_dir)

rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
label_imgs = np.zeros((N, 480, 640), dtype=np.uint8)

all_seg_masks = np.zeros((N, 480, 640))

for i, img_file in enumerate(example_images_dir):
    d = np.load(f'./data/{img_file}', allow_pickle=True, encoding='bytes').item()

    # RGB
    rgb_img = d['rgb']
    rgb_imgs[i] = data_augmentation.standardize_image(rgb_img)

    # XYZ
    xyz_imgs[i] = d['xyz']

    # Label
    label_imgs[i] = np.zeros([480, 640])#d['label']

for i in range(N//4 + 1):
    batch = {
        'rgb' : data_augmentation.array_to_tensor(rgb_imgs[i*4:min((i+1)*4, N-1),...]),
        'xyz' : data_augmentation.array_to_tensor(xyz_imgs[i*4:min((i+1)*4, N-1),...]),
    }

    fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)

    seg_masks = seg_masks.cpu().numpy()

    for j in range(seg_masks.shape[0]):

        #num_objs = max(np.unique(seg_masks[j,...]).max(), np.unique(label_imgs[j,...]).max()) + 1

        #seg_mask_plot = util_.get_color_mask(seg_masks[j,...], nc=num_objs)
        #gt_masks = util_.get_color_mask(label_imgs[j,...], nc=num_objs)

        all_seg_masks[i*4+j,...] = seg_masks[j,...]

with open('./mask.npy', 'wb') as f:
    np.save(f, all_seg_masks)








print("Number of images: {0}".format(N))

### Compute segmentation masks ###
st_time = time()
fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
total_time = time() - st_time
print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

# Get results in numpy
seg_masks = seg_masks.cpu().numpy()
fg_masks = fg_masks.cpu().numpy()
center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
initial_masks = initial_masks.cpu().numpy()

rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
total_subplots = 6

fig_index = 1
for i in range(1):

    #print(i)
    num_objs = max(np.unique(seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1

    rgb = rgb_imgs[i].astype(np.uint8)
    depth = xyz_imgs[i,...,2]
    seg_mask_plot = util_.get_color_mask(seg_masks[i,...], nc=num_objs)
    gt_masks = util_.get_color_mask(label_imgs[i,...], nc=num_objs)

    images = [rgb, depth, seg_mask_plot, gt_masks]
    titles = [f'Image {i+1}', 'Depth',
              f"Refined Masks. #objects: {np.unique(seg_masks[i,...]).shape[0]-1}",
              f"Ground Truth. #objects: {np.unique(label_imgs[i,...]).shape[0]-1}"
             ]
    util_.subplotter(images, titles, fig_num=i+1)

    # Run evaluation metric
    eval_metrics = evaluation.multilabel_metrics(seg_masks[i,...], label_imgs[i])
    print(f"Image {i+1} Metrics:")
    print(eval_metrics)

plt.show()
'''

x, y = np.where(seg_mask_plot[...,2] > 0)
xyz = np.squeeze(xyz_img)

xyz = xyz[x,y]
rgb = rgb_img[x,y]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb/255.0)
#x =
'''