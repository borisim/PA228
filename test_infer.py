import sys
# import matplotlib.pyplot as plt
# from pa228_tools import ishow
from pa228_tools import ishow
import torch
import pandas as pd
# from torchview import draw_graph
# from network import ModelExample
from dataset import SampleDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from network import SampleModel
from pa228_tools import  plot_seg_result, show_seg_sample
import numpy as np

import glob

# dataset_path = "/home/borisim/Documents/school/pa228/project/Project_Template/PROJCODE_UCO/data"
dataset_path = 'data'
PATH = Path('{}'.format(dataset_path), 'data_seg_public')
img_dir = PATH / 'img'
mask_dir = PATH / 'mask'
img_files = glob.glob("{}/*/*.png".format(img_dir))
mask_files = glob.glob("{}/*/*.png".format(mask_dir))
print(len(img_files))
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
traindataset, valdataset = SampleDataset(train_df, transforms), SampleDataset(valid_df, transforms)





# model_path = 'model.pt'  # an example of model_path parameter
# model = torch.load(model_path)
# model.eval()

model = SampleModel(num_class=8)
IDX = 2285
x, y = traindataset[IDX]
print('ahoj')
ishow(y)
pred = model(x.unsqueeze(0))
# torch.save(model, 'model.pt')

# print(x.dtype())
# ishow(np.transpose(x.numpy(), (1, 2, 0)))
# show_seg_sample((x,y))
plot_seg_result(pred)