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

import uois.src.data_augmentation as data_augmentation
import uois.src.segmentation as segmentation
import uois.src.evaluation as evaluation
import uois.src.util.utilities as util_
import uois.src.util.flowlib as flowlib

np.random.seed(0)
torch.manual_seed(1)

def generate_segmentation_masks(input_dir, output_dir, plot=False):

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

    curr_dir = os.getcwd()

    checkpoint_dir = curr_dir + '\\uois\\models\\'

    dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
    rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'

    uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
    uois_net_3d = segmentation.UOISNet3D(uois3d_config,
                                        dsn_filename,
                                        dsn_config,
                                        rrn_filename,
                                        rrn_config
                                        )

    example_images_dir = os.listdir(input_dir)

    N = len(example_images_dir)


    all_rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    all_xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)


    label_imgs = np.zeros((N, 480, 640), dtype=np.uint8)

    all_seg_masks = np.zeros((N, 480, 640))

    for i, img_file in enumerate(example_images_dir):
        d = np.load(f'{input_dir}/{img_file}', allow_pickle=True, encoding='bytes').item()

        print(img_file)

        # RGB
        rgb_img = d['rgb']
        all_rgb_imgs[i] = data_augmentation.standardize_image(rgb_img)

        # XYZ
        all_xyz_imgs[i] = d['xyz']

        #print(d['xyz'])

        # Label
        label_imgs[i] = np.zeros([480, 640])


    for i in range(N//4 + 1):

        batch = {
            'rgb' : data_augmentation.array_to_tensor(all_rgb_imgs[i*4:(i+1)*4,...]),
            'xyz' : data_augmentation.array_to_tensor(all_xyz_imgs[i*4:(i+1)*4,...]),
        }

        fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)

        seg_masks = seg_masks.cpu().numpy()

        for j in range(seg_masks.shape[0]):

            all_seg_masks[i*4+j,...] = seg_masks[j,...]

    with open(f'{output_dir}/mask.npy', 'wb') as f:
        np.save(f, all_seg_masks)

    if plot:
        for i in range(N):

            num_objs = max(np.unique(all_seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1

            num_objs = int(num_objs)

            print(num_objs)
            if num_objs < 2:
                print('Nothing was detected!')
                continue


            rgb = all_rgb_imgs[i].astype(np.uint8)
            depth = all_xyz_imgs[i,...,2]
            seg_mask_plot = util_.get_color_mask(all_seg_masks[i,...], nc=num_objs)

            images = [rgb, depth, seg_mask_plot]
            titles = [f'Image {i+1}', 'Depth',
                    f"Refined Masks. #objects: {np.unique(all_seg_masks[i,...]).shape[0]-1}",
                    ]
            util_.subplotter(images, titles, fig_num=i+1)

            eval_metrics = evaluation.multilabel_metrics(all_seg_masks[i,...], label_imgs[i])

            print('hello there')
            plt.savefig('test.png')

if __name__ == '__main__':

    generate_segmentation_masks('./data/pc', './data/segmentation_masks', plot=True)
