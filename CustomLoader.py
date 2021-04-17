import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import torch
import numpy as np
import json

class ImageLoader(Dataset):
    def __init__(self, json_file, root_dir, transform = None):
        self.annotations = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return(len(self.annotations))
        
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.from_numpy(np.asarray(self.annotations.iloc[index,1]))
        
        if self.transform:
            image = self.transform(image)
        
        return image, y_label
    
    
class MultiModalLoader(Dataset):
    def __init__(self, json_file, root_dir, transform = None):
        self.annotations = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return(len(self.annotations))
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        
        text = self.annotations.iloc[index,1]        
        
        hate = torch.from_numpy(np.asarray(self.annotations.iloc[index, 2]))
        
        sent = torch.from_numpy(np.asarray(self.annotations.iloc[index, 2]))
        
        y_label = torch.from_numpy(np.asarray(self.annotations.iloc[index,4]))
        
        if self.transform:
            image = self.transform(image)
        
        return image, text, hate, sent, y_label