from utils.Calculator import CAlculator
import os
import sys
import time
import torch
from torch import optim
import numpy as np
from  models.mfanet import MFANet
from utils.dataset import CrowdDataset
from losses.nel import NEL_Loss


class  MFANetTrainer():
    def __init__(self, args):
        self.args = args
    def setup(self):
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.set_device(args.device)
            print('using gpu device:{}'.format(args.device))
        else:
            print("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        train_image_root= os.path.join(args.data_dir,'train_data/images')
        train_dmap_root= os.path.join(args.data_dir,'train_data/ground_truth')
        val_image_root= os.path.join(args.data_dir,'val_data/images')
        val_dmap_root=  os.path.join(args.data_dir,'val_data/ground_truth')

        torch.cuda.manual_seed(args.seed)
        self.model = MFANet()
        self.model.to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),args.lr,alpha=0.9)
        self.criterion = NEL_Loss().to(self.device)
        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
                
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        self.save_dir = args.save_dir
        train_dataset=CrowdDataset(train_image_root,train_dmap_root,gt_downsample = args.downsample_ratio,phase='train')
        self.train_loader=torch.utils.data.DataLoader(train_dataset,batch_size = args.batch_size,shuffle=True)
        val_dataset=CrowdDataset(val_image_root,val_dmap_root,gt_downsample = args.downsample_ratio,phase='val')
        self.val_loader=torch.utils.data.DataLoader(val_dataset,batch_size = args.batch_size,shuffle=False)        
        self.best_mae = np.inf
        self.best_mse = np.inf  
        self.best_epoch = np.inf
        print("setup successfully!")
        
    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            print('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = CAlculator()
        epoch_start = time.time()
        self.model.train() 
        
        for step, (img,gt_dmap) in enumerate(self.train_loader):
            #print(step)
            width = int(img.shape[2]/2)
            height = int(img.shape[3]/2)
 
            image_left,image_right = torch.chunk(img,chunks = 2,dim = 3)
            image_0,image_1 = torch.chunk(image_left,chunks = 2,dim = 2)
            image_2,image_3 = torch.chunk(image_right,chunks = 2,dim = 2)
            
            map_left,map_right = torch.chunk(gt_dmap,chunks = 2,dim = 3)
            map_0,map_1 = torch.chunk(map_left,chunks = 2,dim = 2)
            map_2,map_3 = torch.chunk(map_right,chunks = 2,dim = 2)
            img1 = torch.cat((image_0,image_1,image_2,image_3),0)
            gt_dmap1 = torch.cat((map_0,map_1,map_2,map_3),0)

            for chunkid in range(0):
               index = np.random.randint(0,width-1,size=1)[0]
               #print(index)
               indey = np.random.randint(0,height-1,size=1)[0]
               sub_image = img[:,:,index:index+width,indey:indey+height]
               sub_map = gt_dmap[:,:,int(index/8):int((index+width)/8),int(indey/8):int((indey+height)/8)]
        
               img1 = torch.cat((img1,sub_image),0)
               gt_dmap1 = torch.cat((gt_dmap1,sub_map),0)
            #print(img1.shape)
            
            with torch.set_grad_enabled(True):
                img=img1.to(self.device)
                gt_dmap=gt_dmap1.to(self.device)
                #forward propagation
                et_dmap = self.model(img)
                # calculate loss
                loss=self.criterion(et_dmap,gt_dmap)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pre_count = torch.sum(et_dmap).detach().cpu().numpy()
                gd_count = torch.sum(gt_dmap).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item())

        print('Epoch {} Train, Loss: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(),time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_epoch.tar'.format(1))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval() 
        epoch_res = []
        mae = 0
        mse = 0
        # Iterate over data.
        for i,(img,gt_dmap) in enumerate(self.val_loader):
            img=img.to(self.device)
            gt_dmap1=gt_dmap.to(self.device)
            # forward propagation
            et_dmap=self.model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap1.data.sum()).item()
            mse+=np.square((et_dmap.data.sum()-gt_dmap1.data.sum()).item())
            del img,gt_dmap,et_dmap
            
        mse = np.sqrt(mse/len(self.val_loader))
        mae = mae/len(self.val_loader)
        print('Epoch {} Test, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if  mae  <  self.best_mae:
            self.best_mse = mse
            self.best_mae = mae

            self.best_epoch = self.epoch
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        print("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                      self.best_mae,
                                                                      self.best_epoch))


