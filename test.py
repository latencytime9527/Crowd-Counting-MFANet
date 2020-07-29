import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import numpy as np
import logging
from models.mfanet import MFANet
from utils.dataset import CrowdDataset

def cal_mae(img_root,gt_dmap_root,model_param_path):
    device=torch.device("cuda")
    model = MFANet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    dataset=CrowdDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    mse=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            mse+=np.square((et_dmap.data.sum()-gt_dmap.data.sum()).item())
            del img,gt_dmap,et_dmap
        mse = np.sqrt(mse/len(dataloader))
        mae = mae/len(dataloader)
        logging.info('min_mae = {}, min_mse = {}'.format(mae,mse))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    device=torch.device("cuda")
    model = MFANet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            plt.show()
            break


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='/home/sda_disk/sda_disk/data/ShanghaiTech_Dataset/part_B_final/test_data/images'
    gt_dmap_root='/home/sda_disk/sda_disk/data/ShanghaiTech_Dataset/part_B_final/test_data/ground_truth'
    model_param_path='./checkpoints/epoch_best_partB.pth'
    torch.cuda.set_device(1)
    print("using device:",1) 
    cal_mae(img_root,gt_dmap_root,model_param_path)
    estimate_density_map(img_root,gt_dmap_root,model_param_path,45) 
