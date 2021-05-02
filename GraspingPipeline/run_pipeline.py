import os
import numpy as np

from ros import process_ros_data
from get_segmentation_masks_uois import generate_segmentation_masks
from get_processed_pc_uois import process_pc
#from test_data import run


class Depthmap:

    def __init__(

        self, process,
        input_dir = './data/sensor_data', output_dir = './data/pc'):

        self.process_func = process
        self.input_dir = input_dir
        self.output_dir = output_dir

    def process(self):
        '''
        for image in os.listdir(self.input_dir):
            print(f'{self.input_dir}/{image}')

            try:
                image_and_pc = self.process_func(f'{self.input_dir}/{image}', plot=True)
            except Exception as e:
                print(f'Warning: {image} errored out with exception {e}. Will continue onto next artifact')
                continue


            image_name = '.'.join(image.split('.')[:-1]) + '.npy'
        '''

        image_name = 'rgbxyz_ros.npy'
        image_and_pc = process_ros_data()

        with open(f'{self.output_dir}/{image_name}', 'wb') as f:
            np.save(f, image_and_pc)

if __name__ == '__main__':

    intel_depthmap = Depthmap(process_ros_data)
    intel_depthmap.process()

    generate_segmentation_masks('./data/pc', './data/segmentation_masks', plot=True)

    items = os.listdir('./pytorch_6dof-graspnet/demo/data')

    for item in items:
        os.remove('./pytorch_6dof-graspnet/demo/data' + f'/{item}')

    process_pc(
        input_dir='./data/pc',
        masks_filepath='./data/segmentation_masks/mask.npy',
        output_dir='./pytorch_6dof-graspnet/demo/data',
        plot=False
    )

    #run()


