import os
import cv2
import torch
from PIL import Image
import numpy as np
import pickle
from torch.utils.data import Dataset

class IndoorDataset(Dataset):
    def __init__(self, data_dir, dataset='', transform=None, zero_shot=False):
        self.pics = sorted(os.listdir(data_dir))
        self.data_dir = data_dir
        self.transform = transform
        self.zero_shot = zero_shot
        if self.zero_shot:
            self.dataset = dataset
    
    def __len__(self):
        return int(len(self.pics) / 4)
    
    def __getitem__(self, idx):
        file_paths = self.pics[idx*4:(idx+1)*4]
        
        all_data = []
        all_label = []
        for count, file_path in enumerate(file_paths):
            temp = Image.open(os.path.join(self.data_dir, file_path))
            data = temp.copy()
            temp.close()
            
            file_name = os.path.splitext(file_path)[0]
            if self.zero_shot:
                label = int(file_name.split('_')[0]) 
                with open('data/' + self.dataset + '/all_loc_to_dict.pkl', 'rb') as f:
                    loc_to_dict = pickle.load(f)
                label = torch.LongTensor((loc_to_dict[label] - 1,)) # minus 1 to 0-index labeling [0, 213 or 393]
            else:
                label = torch.LongTensor((int(file_name.split('_')[0]) - 1,)) # minus 1 to 0-index labeling [0, 213]

            if self.transform:
                data = self.transform(data)
            
            all_data.append(data.unsqueeze(0))
            all_label.append(label)
        
        data = torch.cat(all_data)
        label = torch.cat(all_label)

        return data, label


class MapDataset(Dataset):
    def __init__(self, args):
        with open('data/' + args.dataset + '/loc_vec.npy', 'rb') as f:
            self.loc_vec = np.load(f)[1:, :]
    
    def __len__(self):
        return len(self.loc_vec)
    
    def __getitem__(self, idx):
        return torch.FloatTensor((self.loc_vec[idx])), torch.LongTensor((idx,))