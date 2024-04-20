# STUDENT's UCO: 000000

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.
import torch
from torch.utils.data import Dataset
from skimage import color, io

class SampleDataset(Dataset):

    def __init__(self,
                 dataset_df,
                 transforms=None,
                 ):

        self.samples_df = dataset_df     # dataframe of all samples
        self.transform = transforms 

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        sample = self.samples_df.iloc[idx]

        img = io.imread(sample['img']) 
        ann = io.imread(sample['mask'])

        transformed = self.transform(image=img, mask=ann)
        res_img = transformed['image']
        res_mask = (transformed['mask'] - 1).type(torch.long)
        
        return res_img, res_mask  


# class OxfordIIITPetDataset(Dataset):
    
#     def __init__(self,
#                  dataset_df,
#                  transforms,
#                  ):

#         self.samples_df = dataset_df     # dataframe of all samples
#         self.transform = transforms      # list of transforms
        
#     def __len__(self):
#         return len(self.samples_df)
    
#     def __getitem__(self, index):
        
#         # get sample atributes
#         sample = self.samples_df.iloc[index]
        
#         # read images
#         img = io.imread(sample['img_path']) #.astype(np.uint8)
#         ann = io.imread(sample['ann_path'])
                
#         # transform img colorspaces
#         if len(img.shape) == 2:
#             img = gray2rgb(img)
#         if img.shape[-1] == 4:
#             img = rgba2rgb(img)
         
#         # debug
#         # print(img.shape, img.dtype, sample['img_path'])
            
#         # apply transformations
#         transformed = self.transform(image=img, mask=ann)
#         res_img = transformed['image']
#         res_mask = (transformed['mask'] - 1).type(torch.long)
        
#         return res_img, res_mask  