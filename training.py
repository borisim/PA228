# STUDENT's UCO: 519192

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <path_2_dataset>

import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchview import draw_graph
from network import UNet
from dataset import SEGDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


# sample function for model architecture visualization
# saves visualization of model architecture to the model_architecture.png
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(network):
    model_graph = draw_graph(network, input_size=(1,3,512,1024), graph_dir='LR', save_graph=True, filename="model_architecture")


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


def loss_batch(model, loss_func, xb, yb, dev, opt=None):
    xb, yb = xb.to(dev), yb.to(dev)
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def train(model, train_dl, loss_func, dev, opt):
        model.train()
        loss, size = 0, 0
        for b_idx, (xb, yb) in tqdm(enumerate(train_dl), total=len(train_dl), leave=False): # tqdm didnt work, hence the print
            b_loss, b_size = loss_batch(model, loss_func, xb, yb, dev, opt)
            loss += b_loss * b_size
            size += b_size
            # print(b_idx)
            
        return loss / size
    
    
def validate(model, valid_dl, loss_func, dev, opt=None):
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, dev) for xb, yb in valid_dl]
            )
            
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def fit(net, epochs, trainloader, validloader, loss_fn, optimizer, device):
    train_losses = []
    validation_losses = []

    best = 3
    for epoch in tqdm(range(epochs), 'epochs'):
        # print('training')
        loss = train(net, trainloader, loss_fn, device, optimizer)
        # print('validating')
        val_loss = validate(net, validloader, loss_fn, device)

        train_losses.append(loss)
        validation_losses.append(val_loss)
        print(f'epoch {epoch+1}/{epochs}, loss: {loss : .05f}, validation loss: {val_loss:.05f}')

        if loss < best:
            best = loss
            torch.save(net, 'model.pt')

    print('Training finished!')
    return train_losses, validation_losses


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
    'epoch': 0,
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

    # train_df, test_df = train_test_split(df, test_size=.3, random_state=1)
    # train_df, valid_df = train_test_split(train_df, test_size=.3, random_state=2)

    transforms = A.Compose([
                            A.HorizontalFlip(p=0.5),  
                            A.GaussianBlur(blur_limit=(3, 7), p=0.5),  
                            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  
                            # A.ShiftScaleRotate(rotate_limit=30, p=1, border_mode=0, value=0, mask_value=3),
                            
                            A.Normalize(mean=(0.3210, 0.2343, 0.2740), std=(0.1852, 0.1621, 0.1804)),
                            ToTensorV2(),
                            ]   
                        )

    traindataset = SEGDataset(train_df, transforms=transforms)
    valdataset = SEGDataset(valid_df, transforms=transforms)
    
    trainloader = torch.utils.data.DataLoader(traindataset,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'])

    valloader = torch.utils.data.DataLoader(valdataset,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'])

    net = UNet(config['n_classes'])
    net.to(device)
    draw_network_architecture(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    tr_losses, val_losses = fit(net, config['epoch'], trainloader, valloader, loss_fn, optimizer, device)
    
    plot_learning_curves(tr_losses, val_losses)
    return


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