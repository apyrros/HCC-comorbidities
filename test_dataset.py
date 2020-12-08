import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.labels import *

class TestDataset(Dataset):
    """
    Load images and corresponding labels for
    """
    def __init__(self, data_path, transforms=None, size=256):
        """
        Initialize data set
        Loads and preprocesses data
        @param data_path : path to data and labels
        @param size : size of each xray
        """
        if not os.path.exists(data_path):
            raise IOError('Path given for TestDataset {} does not exist...'.format(data_path))
        self.data_path = data_path
        self.data = os.listdir(data_path)
        self.size = size

        self.transforms = transforms

    def __len__(self):
        """
        Get length of dataset
        @return len : length of dataset
        """
        return len(self.data)

    def __getitem__(self,idx):
        """
        Gets data at a certain index
        @param idx : idx of data desired
        @return xray : xray image at idx
        @return tensor : tensor of condition codes at idx
        """
        fname = self.data[idx]
        xray = Image.open(os.path.join(self.data_path, fname))
        xray = xray.resize((self.size, self.size), Image.LANCZOS)
        xray = xray.convert('L')
        if self.transforms:
            xray = self.transforms(xray)

        return xray

    def at(self,idx):
        """
        Gets directory name for a certain index
        @param idx : idx of data directory desired
        @return name : name of study at idx
        """
        return self.data[idx]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    c = TestDataset('data/test')
    print(len(c))
    print(c[0])
    print(c.at(2))
    data = DataLoader(c, batch_size=4, shuffle=True)
    for s in data:
        print(s[0].shape, s[0], s[0].shape)
        print(s[1].shape, s[1], s[1].shape)
        print(s[2].shape, s[2], s[2].shape)
        print(s[3].shape, s[3], s[3].shape)
        break
