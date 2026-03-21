import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
         
        self.transforms = transforms
        if train:
          file_paths = [f'data_batch_{i}' for i in range(1,6)]
        else:
          file_paths = ['test_batch']
        X = []
        Y = []
        for sub_path in file_paths:
          with open(os.path.join(base_folder, sub_path), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            X.append(dict[b'data'])
            Y.append(dict[b'labels'])
        X = np.concatenate(X, axis = 0)
        X = X/255
        X = X.reshape((-1,3,32,32))
        Y = np.concatenate(Y, axis=None)
        self.X = X
        self.Y = Y
         

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
         
        return (self.X[index], self.Y[index])
         

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
         
        return self.Y.size
         