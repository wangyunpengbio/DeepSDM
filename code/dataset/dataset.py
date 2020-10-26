import os

import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensor
import albumentations

# transform里面可以加个validation time augmentation
def generate_transforms(image_size):
    train_transform = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.OneOf([
            albumentations.RandomContrast(),
            albumentations.RandomGamma(),
            albumentations.RandomBrightness(),
             ], p=0.3),
        albumentations.OneOf([
            albumentations.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albumentations.GridDistortion(),
            albumentations.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
        albumentations.OneOf([
        albumentations.RandomSizedCrop(min_max_height=(int(image_size*0.8), image_size), height=image_size, width=image_size,p=0.5),
        albumentations.ShiftScaleRotate(rotate_limit=30,p=0.5),
        albumentations.RandomCrop(height=int(image_size*0.8), width=int(image_size*0.8),p=0.2),
        albumentations.RandomCrop(height=int(image_size*0.7), width=int(image_size*0.7),p=0.2),
        ], p=0.8),
        albumentations.PadIfNeeded(min_height=image_size, min_width=image_size,border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ],p=1)


    val_transform = albumentations.Compose([
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ],p=1)
    
    return train_transform, val_transform

class PneumothoraxDataset(Dataset):
    def __init__(self, data_folder, mode, transform=None,
                 fold_index=None, folds_distr_path=None):
        
        self.transform = transform
        self.mode = mode
        
        # change to your path
        self.train_image_path = '{}/train/'.format(data_folder)
        self.train_mask_path = '{}/mask/'.format(data_folder)
        self.test_image_path = '{}/test/'.format(data_folder)
        
        self.fold_index = None
        self.folds_distr_path = folds_distr_path
        self.set_mode(mode, fold_index)
        self.to_tensor = ToTensor()

    def set_mode(self, mode, fold_index):
        if fold_index != None:
            fold_index = str(fold_index)
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            folds = pd.read_csv(self.folds_distr_path)
            folds.fold = folds.fold.astype(str)
            folds = folds[folds.fold != fold_index]
            
            self.train_list = folds.fileID.values.tolist()
            self.pne_num = folds.pne_num.values.tolist()

            self.num_data = len(self.train_list)

        elif self.mode == 'val' or self.mode == 'cross-val':
            folds = pd.read_csv(self.folds_distr_path)
            folds.fold = folds.fold.astype(str)
            folds = folds[folds.fold == fold_index]
            
            self.val_list = folds.fileID.values.tolist()
            self.pne_num = folds.pne_num.values.tolist()
            
            self.num_data = len(self.val_list)

        elif self.mode == 'test':
            self.test_list = sorted(os.listdir(self.test_image_path))
            self.num_data = len(self.test_list)

    def __getitem__(self, index):
        if self.fold_index is None and self.mode != 'test':
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return
        
        if self.mode == 'test':
            image = cv2.imread(os.path.join(self.test_image_path, self.test_list[index]), 1)
            if self.transform:
                sample = {"image": image}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image = sample['image']
            image_id = self.test_list[index].replace('.png', '')
            return image_id, image
        
        elif self.mode == 'cross-val':
            image = cv2.imread(os.path.join(self.train_image_path, self.val_list[index] + ".png"), 1)
            if self.pne_num[index] == 0:
                mask = np.zeros((1024, 1024))
                cls_label = torch.FloatTensor([0])
            else:
                mask = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index] + ".png"), 0)
                cls_label = torch.FloatTensor([1])
            if self.transform:
                sample = {"image": image, "mask": mask}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image, mask = sample['image'], sample['mask']
            image_id = self.val_list[index].replace('.png', '')
            return image_id, image, mask, cls_label

        elif self.mode == 'train':
            image = cv2.imread(os.path.join(self.train_image_path, self.train_list[index] + ".png"), 1)
            if self.pne_num[index] == 0:
                mask = np.zeros((1024, 1024))
                cls_label = torch.FloatTensor([0])
            else:
                mask = cv2.imread(os.path.join(self.train_mask_path, self.train_list[index] + ".png"), 0)
                cls_label = torch.FloatTensor([1])

        elif self.mode == 'val':
            image = cv2.imread(os.path.join(self.train_image_path, self.val_list[index] + ".png"), 1)
            if self.pne_num[index] == 0:
                mask = np.zeros((1024, 1024))
                cls_label = torch.FloatTensor([0])
            else:
                mask = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index] + ".png"), 0)
                cls_label = torch.FloatTensor([1])
            
        if self.transform:
            sample = {"image": image, "mask": mask}
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
            image, mask = sample['image'], sample['mask']
        return image, mask, cls_label
         
    def __len__(self):
        return self.num_data


from torch.utils.data.sampler import Sampler
class PneumoSampler(Sampler):
    def __init__(self, folds_distr_path, fold_index, demand_non_empty_proba, mode):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba
        
        self.folds = pd.read_csv(folds_distr_path)
        self.folds.fold = self.folds.fold.astype(str)
        if fold_index != None:
            fold_index = str(fold_index)
        if mode == 'train':
            self.folds = self.folds[self.folds.fold != fold_index].reset_index(drop=True)
        elif mode == 'val':
            self.folds = self.folds[self.folds.fold == fold_index].reset_index(drop=True)
        else:
            print('!!!!!!mode is WRONG!!!!!!')
            return

        self.positive_idxs = self.folds[self.folds.pne_num != 0].index.values
        self.negative_idxs = self.folds[self.folds.pne_num == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        
    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
