import numpy as np
import os
from matplotlib import image



images = os.listdir('./data')

for im in images:

    with open(f'./data/{im}', 'rb') as f:

        pic = np.load(f, allow_pickle=True).item()['rgb']

        image.imsave(f'{im}.png', pic)

