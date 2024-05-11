# STUDENT's UCO: 000000

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <path_2_dataset>

import sys
import matplotlib.pyplot as plt

import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
# from torchview import draw_graph
from network import UNet
from dataset import SampleDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from pa228_tools import train, validate
import glob
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2



# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(network, input_sample):
    # saves visualization of model architecture to the model_architecture.png
    model_graph = draw_graph(network, input_sample, graph_dir='LR', save_graph=True, filename="model_architecture")


# sample function for losses visualization
def plot_learning_curves(train_losses, validation_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


# sample function for training
def fit(net, batch_size, epochs, trainloader, validloader, loss_fn, optimizer, device):
    train_losses = []
    validation_losses = []

    worst = 3
    for epoch in tqdm(range(epochs), 'epochs'):
        print('training')
        loss = train(net, trainloader, loss_fn, device, optimizer)
        print('validating')
        val_loss = validate(net, validloader, loss_fn, device)

        train_losses.append(loss)
        validation_losses.append(val_loss)
        print(f'epoch {epoch+1}/{epochs}, loss: {loss : .05f}, validation loss: {val_loss:.05f}')

        if loss < worst:
            worst = loss
            torch.save(net, 'model.pt'.format(epoch))

        plot_learning_curves(train_losses, validation_losses)
      
    print('Training finished!')
    return train_losses, validation_losses


# declaration for this function should not be changed
def training(dataset_path):
    """
    training(dataset_path) performs training on the given dataset;
    saves:
    - model.pt (trained model)
    - learning_curves.png (learning curves generated during training)
    - model_architecture.png (a scheme of model's architecture)

    Parameters:
    - dataset_path (string): path to a dataset

    Returns:
    - None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    config = {
    'batch_size': 5,
    'epoch': 40,
    'num_workers': 1,
    'dropout': 0.5,
    'lr': 0.0001,
    'n_classes': 8
    }

    PATH = Path('{}'.format(dataset_path), 'data_seg_public')
    img_dir = PATH / 'img' 
    mask_dir = PATH / 'mask' 
    img_files = sorted(glob.glob("{}/*/*.png".format(img_dir))) 
    mask_files = sorted(glob.glob("{}/*/*.png".format(mask_dir)))
    df = pd.DataFrame({'img': img_files, 'mask': mask_files})
    train_df, valid_df = train_test_split(df, test_size=.2, random_state=2)

    transforms = A.Compose([
                            A.HorizontalFlip(p=0.5),  
                            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  
                            A.GaussianBlur(blur_limit=(3, 7), p=0.5),  

                            A.Normalize(mean=(0.3210, 0.2343, 0.2740), std=(0.1852, 0.1621, 0.1804)),
                            ToTensorV2(),
                            ]   
                        )

    traindataset = SampleDataset(train_df, transforms=transforms)
    valdataset = SampleDataset(valid_df, transforms=transforms)
    
    trainloader = torch.utils.data.DataLoader(traindataset,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'])

    valloader = torch.utils.data.DataLoader(valdataset,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'])


    class SoftDiceLoss(nn.Module):
        def __init__(self):
            super(SoftDiceLoss, self).__init__()

        def forward(self, input, target):
            smooth = 1

            input_flat = input.flatten(2).sigmoid()
            target_flat = target.flatten(2)
            
            intersection = torch.sum(input_flat * target_flat, dim=-1)
            union = input_flat.sum(dim=-1) + target_flat.sum(dim=-1)
            dice_coeff = (2. * intersection + smooth) / (union + smooth)

            return 1 - dice_coeff.mean()

    
    net = UNet(config['n_classes'])
    net.to(device)
    # loss_fn = SoftDiceLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    tr_losses, val_losses = fit(net, config['batch_size'], config['epoch'], trainloader, valloader, loss_fn, optimizer, device)
    
    # draw_network_architecture(net, input_sample)
    plot_learning_curves(tr_losses, val_losses)
    return

    # 6h+, 30 epoch, 0.63%
    # new02 0.63%
    # new01 0.63%
    # ce026 0.40%



# #### code below should not be changed ############################################################################
def get_arguments():
    if len(sys.argv) != 2:
        print("Usage: python training.py <path_2_dataset> ")
        sys.exit(1)

    try:
        path = sys.argv[1]
    except Exception as e:
        print(e)
        sys.exit(1)
    return path


if __name__ == "__main__":
    path_2_dataset = get_arguments()
    training(path_2_dataset)