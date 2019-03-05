import os
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, dir_imgs, dir_masks, img_size=(720, 1280), Transform=False):
        
        self.dir_imgs = dir_imgs
        self.img_size = img_size
        self.dir_masks = dir_masks
        self.Transform = Transform
        
        # List of images names
        self.image_arr = os.listdir(self.dir_imgs)
        # List of images masks
        self.mask_arr = os.listdir(self.dir_masks)
        # Transforms
        self.to_tensor = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize(img_size)
        # Dataset lenght
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):

        # Define one imege and mask
        single_image_name = self.image_arr[index]
        single_image_mask_name = self.mask_arr[index]
        
        # Open image and mask
        img_as_img = Image.open(os.path.join(self.dir_imgs, single_image_name))
        mask_as_img = Image.open(os.path.join(self.dir_masks, single_image_mask_name))

        if self.Transform:
            # Transform image and mask to tensor
            img_as_img = self.resize(img_as_img)
            mask_as_img = self.resize(mask_as_img)

        img_as_tensor = self.to_tensor(img_as_img)
        mask_as_tensor = self.to_tensor(mask_as_img)

        return (img_as_tensor, mask_as_tensor)

    def __len__(self):
        # Total dataset size
        return self.data_len