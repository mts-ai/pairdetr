import cv2
from typing import Dict, List, Any
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from config import CFG
import pytorch_lightning as pl
import transforms as T
from PIL import Image
class CrowdHuman_Dataset(pl.LightningDataModule):
    def __init__(self ,train_df: Dict[str, Any], transforms=None):
        self.image_ids = np.asarray(list(train_df.keys()))
        self.df = train_df
        self.transforms = transforms
        
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def __getitem__(self,index: int)  -> tuple:
        image_id = self.image_ids[index]
        records = self.df[image_id]
        if len(records) != 0:
            labels = np.asarray(records)[:, 0]
        else:
            labels = []
        image = cv2.imread(f'{CFG.image_path}/{image_id.split(".")[0]}.jpg')
        height, width,_ = image.shape
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            
        if len(records) != 0:
            boxes1 = np.asarray(records)[:, 1:5]
            boxes2 = np.asarray(records)[:, 5:]
            area = boxes1[:,2]*boxes1[:,3] + boxes2[:,2]*boxes2[:,3]
            area = torch.as_tensor(area, dtype = torch.float32)     
            boxes = np.concatenate((boxes2, boxes1), axis = 0)
            boxes[:,:2] -= (boxes[:,2:]/2)
            boxes[:,2:] += boxes[:,:2]
            boxes[:,0] *= width
            boxes[:,1] *= height
            boxes[:,2] *= width
            boxes[:,3] *= height
        else:
            boxes = []
            area = []
            area = torch.as_tensor(area, dtype = torch.float32)
        target = {}
        target["boxes"] = torch.as_tensor(boxes,dtype = torch.float32)
        target["labels"] = torch.as_tensor(np.concatenate((labels , labels), axis = 0),dtype = torch.long)
        target["image_id"] = torch.as_tensor(index)
        target["area"] = area
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])
        if self.transforms:            
            image, target = self.transforms(image, target)
    

        temp = target["boxes"]#torch.as_tensor(boxes,dtype = torch.float32)
        labels = target["labels"]
        temp = torch.concat((temp[:len(temp)//2], temp[len(temp)//2:]), dim = 1)
        target['boxes'] = temp
        target['labels'] = torch.as_tensor(labels[:len(labels)//2],dtype = torch.long)
       
        return image, target, image_id
        
def get_train_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#([103.530, 116.280, 123.675], [57.375, 57.120, 58.395])#
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=1400),
        # T.RandomSelect(
        #     T.RandomResize(scales, max_size=1333),
        #     T.Compose([
        #         T.RandomResize([400, 500, 600]),
        #         T.RandomSizeCrop(384, 600),
        #         T.RandomResize(scales, max_size=1333),
        #     ])
        # ),
        normalize,
    ])     

def get_valid_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#([103.530, 116.280, 123.675], [57.375, 57.120, 58.395])#
    ])
    return T.Compose([
            T.RandomResize([800], max_size=1400),
            normalize,
        ])        

