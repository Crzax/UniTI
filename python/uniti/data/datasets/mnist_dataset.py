from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
         
        self.transforms = transforms
        with gzip.open(image_filename) as img:
          magic_num, img_num, row, col = struct.unpack(">4i", img.read(16))
          assert(magic_num == 2051)
          tot_pixels = row * col
          imgs = [np.array(struct.unpack(f"{tot_pixels}B",
                                        img.read(tot_pixels)),
                                        dtype=np.float32)
                  for _ in range(img_num)]
          X = np.vstack(imgs)
          X -= np.min(X)
          X /= np.max(X)
          self.X = X

        with gzip.open(label_filename, 'rb') as lbs:
          magic, label_num = struct.unpack('>2i', lbs.read(8))
          assert(magic==2049)
          self.y = np.array(struct.unpack(f"{label_num}B",
                                        lbs.read(label_num)),
                                        dtype=np.uint8)
         

    def __getitem__(self, index) -> object:
         
        imgs = self.X[index]
        labels = self.y[index]
        if len(imgs.shape) > 1:
          imgs = np.array([self.apply_transforms(img.reshape((28,28,1))) for img in imgs])
        else:
          imgs = self.apply_transforms(imgs.reshape((28,28,1)))
        return (imgs, labels)
         

    def __len__(self) -> int:
         
        return self.X.shape[0]
         