# STUDENT's UCO: 519192

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)

import sys
import os
import numpy as np
import pandas as pd
from dataset import SEGDataset
import torch
from skimage import io
import glob
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# sample function for performing inference for a whole dataset
def infer_all(net, batch_size, dataloader, device):
    # do not calculate the gradients
    with torch.no_grad():
        for i in range(0, 3):
            # generate a random image and save t to output_predictions
            random_image = np.random.rand(50, 50, 3)
            random_image_byte = img_as_ubyte(random_image)
            filename = f"output_predictions/random_image_{i}.png"  # Adjust filename as needed
            io.imsave(filename, random_image_byte)
    return


def plot_pred(pred):
    label_dict = {
         0:(0, 0, 0),
         1:(128, 64, 128),
         2:(70, 70, 70),
         3:(153, 153, 153), 
         4:(107, 142, 35),
         5:(70, 130, 180),
         6:(220, 20, 60),
         7:(0, 0, 142),    
        }

    soft = torch.nn.functional.softmax(pred, dim=1)
    arg = torch.argmax(soft, dim=1).squeeze(0)
    rgb_tensor = torch.zeros(arg.shape[0], arg.shape[1], 3, dtype=torch.uint8)
    colors = torch.tensor([label_dict[i] for i in range(len(label_dict))], dtype=torch.uint8)
    rgb_tensor[:] = colors[arg]

    return rgb_tensor


# declaration for this function should not be changed
def inference(dataset_path, model_path, n_samples):
    """
    inference(dataset_path, model_path='model.pt') performs inference on the given dataset;
    if n_samples is not passed or <= 0, the predictions are performed for all data samples at the dataset_path
    if  n_samples=N, where N is an int positive number, then only N first predictions are performed
    saves:
    - predictions to 'output_predictions' folder

    Parameters:
    - dataset_path (string): path to a dataset
    - model_path (string): path to a model
    - n_samples (int): optional parameter, number of predictions to perform

    Returns:
    - None
    """
  
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()

    PATH = Path('{}'.format(dataset_path))
    img_dir = PATH / 'img'
    mask_dir = PATH / 'mask'
    img_files = sorted(glob.glob("{}/*/*.png".format(img_dir)))
    mask_files = sorted(glob.glob("{}/*/*.png".format(mask_dir)))
    df = pd.DataFrame({'img': img_files, 'mask': mask_files})

    transforms = A.Compose([
                         A.Normalize(mean=(0.3210, 0.2343, 0.2740), std=(0.1852, 0.1621, 0.1804)),
                         ToTensorV2(),
                        ]   
                    )
    traindataset = SEGDataset(df, transforms=transforms)

    n = n_samples if n_samples > 0 else traindataset.__len__()

    for i in range(n):
        x, y = traindataset[i]
        name = df.iloc[i]['img']
        name = os.path.basename(name)
        pred = model(x.unsqueeze(0))
        arg = plot_pred(pred)

        filename = f"output_predictions/{name}".format(name)  
        io.imsave(filename, arg)
        filename = f"output_reference/{name}".format(name)  
        io.imsave(filename, y.numpy().astype(np.uint8))
        print(i)

    return


# #### code below should not be changed ############################################################################
def get_arguments():
    if len(sys.argv) == 3:
        dataset_path = sys.argv[1]
        model_path = sys.argv[2]
        number_of_samples = 0
    elif len(sys.argv) == 4:
        try:
            dataset_path = sys.argv[1]
            model_path = sys.argv[2]
            number_of_samples = int(sys.argv[3])
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print("Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)")
        sys.exit(1)

    return dataset_path, model_path, number_of_samples


if __name__ == "__main__":
    path_2_dataset, path_2_model, n_samples_2_predict = get_arguments()
    inference(path_2_dataset, path_2_model, n_samples_2_predict)
