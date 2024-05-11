from matplotlib import pyplot as plt
import torch
from tqdm.notebook import tqdm
import numpy as np

def ishow(img,
          cmap='viridis',
          title='',
          fig_size=(8,6),
          colorbar=False,
          interpolation='none'):
    ' Function `ishow` displays an image in a new window. '
    
    if img.min() < 0:
        img -= img.min()
        img /= img.max()
    
    extent = (0, img.shape[1], img.shape[0], 0)
    fig, ax = plt.subplots(figsize=fig_size)
    pcm = ax.imshow(img,
              extent=extent,
              cmap=cmap,
              interpolation=interpolation)
    
    ax.set_frame_on(False)
    plt.title(title)
    plt.tight_layout()
    if colorbar:
        
        fig.colorbar(pcm, orientation='vertical')
    plt.show()
    
def pred_loss_prep(xb):
    softmax = torch.nn.functional.softmax(xb, dim=1)
    argmax = torch.argmax(softmax, dim=1)

    binary_masks = torch.zeros_like(xb)
    for class_index in range(xb.shape[1]):
        class_mask = (argmax == class_index).float()
        binary_masks[:, class_index, :, :] = class_mask

    return binary_masks


def loss_batch(model, loss_func, xb, yb, dev, opt=None):
    xb, yb = xb.to(dev), yb.to(dev)
    # loss = loss_func(model(xb), yb)
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def train(model, train_dl, loss_func, dev, opt):
        
        model.train()
        loss, size = 0, 0
        for b_idx, (xb, yb) in tqdm(enumerate(train_dl), total=len(train_dl), leave=False):
            b_loss, b_size = loss_batch(model, loss_func, xb, yb, dev, opt)
            loss += b_loss * b_size
            size += b_size
            
            print(b_idx)
            
        return loss / size
    
    
def validate(model, valid_dl, loss_func, dev, opt=None):
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, dev) for xb, yb in valid_dl]
            )
            
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def plot_seg_result(pred):
    
    # validate the input shape
    assert len(pred.shape) == 4, f'prediction shape {pred.shape} is required to be 4-dimensional'
    
    n_classes = pred.shape[1]
    
    np_pred = pred.detach().cpu().squeeze().numpy()
    images = np.split(np_pred, n_classes, axis=0)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(15, 5))
    
    for i in range(n_classes):
        plot = axes[i].imshow(images[i].squeeze())
        axes[i].set_title(f'prediction - channel {i}')
        # plt.colorbar(plot, ax=axes[i])      
        
    plt.show()


def show_seg_sample(sample):

    img_pt, mask = sample
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    img = img_pt.numpy()
    img = img - img.min()
    img = img / img.max()
    

    ax1.imshow(np.moveaxis(img, 0, -1))
    ax1.set_title('original image')
    ax2.imshow(mask)
    ax2.set_title('GT labels')
    plt.show()


def plot_pred(pred, label):
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
    # ishow(label)


def pred_loss_prep(xb):
    softmax = torch.nn.functional.softmax(xb, dim=1)
    argmax = torch.argmax(softmax, dim=1)

    binary_masks = torch.zeros_like(xb)
    for class_index in range(xb.shape[1]):
        class_mask = (argmax == class_index).float()
        binary_masks[:, class_index, :, :] = class_mask

    return binary_masks


def label_loss_prep(yb): 
    label_dict = {
        (0, 0, 0) : 0,
        (128, 64, 128) : 1,
        (70, 70, 70) : 2,
        (153, 153, 153) : 3, 
        (107, 142, 35) : 4,
        (70, 130, 180) : 5,
        (220, 20, 60) : 6,
        (0, 0, 142) : 7
        }
    
    class_masks = torch.zeros(8, yb.shape[0], yb.shape[1])
    for color, class_index in label_dict.items():
        color_mask = torch.all(yb == torch.tensor(color).view(1, 1, 3), dim=-1).float()
        class_masks[class_index, :, :] = color_mask
    
    return class_masks