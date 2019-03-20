import os
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor, Resize

class UNet_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        
        self.root = root
        self.transform = transform
        self.train = train
        
        if self.train:
            self.dir_imgs = os.path.join(self.root, 'train', 'images')
            self.dir_masks = os.path.join(self.root, 'train', 'masks')
            
            # List of images names
            self.image_arr = os.listdir(self.dir_imgs)
            # List of images masks
            self.mask_arr = os.listdir(self.dir_masks)
        else:
            self.dir_imgs = os.path.join(self.root, 'test', 'images')
            self.dir_masks = os.path.join(self.root, 'test', 'masks')
            
            self.image_arr = os.listdir(self.dir_imgs)
            self.mask_arr = os.listdir(self.dir_masks)

    def __getitem__(self, index):

        # Define one imege and mask
        single_image_name = self.image_arr[index]
        single_image_mask_name = self.mask_arr[index]
        
        # Open image and mask
        img = Image.open(os.path.join(self.dir_imgs, single_image_name))
        mask = Image.open(os.path.join(self.dir_masks, single_image_mask_name))

        if self.transform is not None:
            for transf in self.transform:
                # Transform image and mask
                img = transf(img)
                mask = transf(mask)

        return img, mask

    def __len__(self):
        # Total dataset size
        return len(self.image_arr)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    
        return fmt_str