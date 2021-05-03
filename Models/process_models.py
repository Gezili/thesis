import os
import zipfile

models_to_process = os.listdir('./raw')

print(f'Found {len(models_to_process)} models to process')

for model in models_to_process:

    folder = model.split(".")[0]
    try:
        os.mkdir(f'./processed/{folder}')
    except FileExistsError:
        continue

    with zipfile.ZipFile(f'./raw/{model}') as f:

        with f.open('meshes/model.obj') as fz:
            mesh = fz.read()

            with open(f'./processed/{folder}/model.obj', 'wb') as fr:
                fr.write(mesh)

        with f.open('materials/textures/texture.png') as fz:
            texture = fz.read()

            with open(f'./processed/{folder}/texture.png', 'wb') as fr:
                fr.write(texture)

        with f.open('meshes/model.mtl') as fz:
            mtl = fz.read()

            with open(f'./processed/{folder}/model.mtl', 'wb') as fr:
                fr.write(mtl)


