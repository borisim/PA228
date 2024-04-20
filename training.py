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
from torchview import draw_graph
from network import SampleModel
from dataset import SampleDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from pa228_tools import train, validate
import glob


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

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# sample function for training
def fit(net, batch_size, epochs, trainloader, validloader, loss_fn, optimizer, device):
    train_losses = []
    validation_losses = []

    for epoch in tqdm(range(epochs), 'epochs'):
        loss = train(net, trainloader, loss_fn, device, optimizer)
        val_loss = validate(net, validloader, loss_fn, device)

        train_losses.append(loss)
        validation_losses.append(val_loss)
        print(f'epoch {epoch+1}/{epochs}, loss: {loss : .05f}, validation loss: {val_loss:.05f}')

      
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
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    # config dictionary
    config = {
    'batch_size': 4,
    'epoch': 1,
    'num_workers': 1,
    'dropout': 0.5,
    'lr': 0.0001,
    'optimizer':'Adam',
    'img_size': 128,
    'n_classes': 2
    }

    PATH = Path('{}'.format(dataset_path), 'data_seg_public')
    img_dir = PATH / 'img'
    mask_dir = PATH / 'mask'
    img_files = glob.glob("{}/*/*.png".format(img_dir))
    mask_files = glob.glob("{}/*/*.png".format(mask_dir))
    df = pd.DataFrame({'img': img_files, 'mask': mask_files})

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transforms = A.Compose([
                            A.SmallestMaxSize (512),
                            A.CenterCrop(512, 1024),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),
                            ]   
                        )

    train_df, valid_df = train_test_split(df, test_size=.3, random_state=2)
    traindataset, valdataset = SampleDataset(train_df, transforms=transforms), SampleDataset(valid_df, transforms=transforms)
    
    trainloader = torch.utils.data.DataLoader(traindataset,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'])

    valloader = torch.utils.data.DataLoader(valdataset,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'])


    net = SampleModel()
    # input_sample = torch.zeros((1, 512, 1024))
    # draw_network_architecture(net, input_sample)

    # define optimizer and learning rate
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # train the network for three epochs
    tr_losses, val_losses = fit(net, config['batch_size'], config['epoch'], trainloader, valloader, loss_fn, optimizer, device)

    # save the trained model and plot the losses, feel free to create your own functions
    torch.save(net, 'model.pt')
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
