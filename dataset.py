# STUDENT's UCO: 519192

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.
import torch
from torch.utils.data import Dataset
from skimage import color, io


def bce_lab(y):
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

    class_indices = torch.zeros(y.shape[:2], dtype=torch.uint8)
    for rgb, class_idx in label_dict.items():
        mask = torch.all(y == torch.tensor(rgb), dim=-1)
        class_indices[mask] = class_idx
    
    return class_indices.long()

class SEGDataset(Dataset):
    def __init__(self,
                 dataset_df,
                 transforms=None,
                 ):

        self.samples_df = dataset_df     
        self.transform = transforms 

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        sample = self.samples_df.iloc[idx]
        img = io.imread(sample['img']) 
        ann = io.imread(sample['mask'])

        transformed = self.transform(image=img, mask=ann)
        res_img = (transformed['image']).type(torch.float32) 
        res_mask = (transformed['mask']).type(torch.long)
        res_mask = bce_lab(res_mask)
        
        return res_img, res_mask