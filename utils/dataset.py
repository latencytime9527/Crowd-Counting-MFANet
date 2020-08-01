from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import random
import scipy.io as io
import pandas as pd

class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root,gt_downsample=1,phase='train'):
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_downsample=gt_downsample
        self.phase=phase
        self.img_names=[filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root,filename))]
        self.n_samples=len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))/255
        
        if len(img.shape)==2: 
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)
        
        gt_file = pd.read_csv(os.path.join(self.gt_dmap_root,img_name.replace('.jpg','.csv')),header=None)
        gt_dmap = np.mat(gt_file)
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[:,::-1]
            gt_dmap=gt_dmap[:,::-1]
        
        if self.gt_downsample>1:
            ds_rows=int(img.shape[0]//(self.gt_downsample*2))
            ds_cols=int(img.shape[1]//(self.gt_downsample*2))
            img = cv2.resize(img,(ds_cols*self.gt_downsample*2,ds_rows*self.gt_downsample*2))
            img=img.transpose((2,0,1)) 
            
            gt_dmap=cv2.resize(gt_dmap,(int(ds_cols*self.gt_downsample*2),int(ds_rows*self.gt_downsample*2)))
            gt_dmap=gt_dmap[np.newaxis,:,:]

            img_tensor=torch.tensor(img,dtype=torch.float)
            img_tensor=transforms.functional.normalize(img_tensor,mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)

        return img_tensor,gt_dmap_tensor 


# test code
if __name__=="__main__":
    img_root="/home/UCF_QNRF/train_data/images"
    gt_dmap_root="/home/UCF_QNRF/train_data/ground_truth"
    dataset=CrowdDataset(img_root,gt_dmap_root,gt_downsample=8)
    for i,(img,gt_dmap) in enumerate(dataset):
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(gt_dmap)
        # plt.figure()
        # if i>5:
        #     break
        print(img.shape,gt_dmap.shape)
        print(img.min(),img.max(),gt_dmap.min(),gt_dmap.max())
        break
