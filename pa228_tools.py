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
    

def loss_batch(model, loss_func, xb, yb, dev, opt=None):
    
    xb, yb = xb.to(dev), yb.to(dev)
    pred = model(xb)
    print('got_out')
    yb = yb.argmax(dim=3)
    # print(yb.shape)
    # print(pred.shape)
    print('loss')
    loss = loss_func(pred, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def train(model, train_dl, loss_func, dev, opt):
        
        model.train()
        loss, size = 0, 0
        for b_idx, (xb, yb) in tqdm(enumerate(train_dl), total=len(train_dl), leave=False):
            print('batching')
            b_loss, b_size = loss_batch(model, loss_func, xb, yb, dev, opt)
            print('done batching')

            loss += b_loss * b_size
            size += b_size
            
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
        plt.colorbar(plot, ax=axes[i])      
        
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
# # training step
#         loss = train(model, train_dl, loss_func, dev, opt)
#         # evaluation step
#         val_loss = validate(model, valid_dl, loss_func, dev)

#         # Log metrics with MLflow
#         mlflow.log_metric("training_loss", loss, step=epoch)
#         mlflow.log_metric("validation_loss", val_loss, step=epoch)
        
#         print(f'epoch {epoch+1}/{epochs}, loss: {loss : .05f}, validation loss: {val_loss:.05f}')