import os
import numpy as np 
import pandas as pd 
from datetime import datetime
import time
import random
from tqdm.autonotebook import tqdm
import cv2
import sys
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F
import math
from glob import glob
from typing import Dict, List, Any
import json
import pytorch_lightning as pl
from transformers import DeformableDetrForObjectDetection
sys.path.append('./losses/')
from losses.loss.matcher import HungarianMatcher
from losses.loss.detr import SetCriterion
from dataset import CrowdHuman_Dataset
from train_lightning_d import Detr_light
import shutil
from losses.util import box_ops
from torch.multiprocessing import Queue, Process
import transforms as T
from utils import collate_fn
from tqdm import tqdm
def get_valid_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
            T.RandomResize([800], max_size=1400),
            normalize,
        ])        
class CFG:
    seed = 42
    num_classes = 3
    num_queries = 1500
    null_class_coef = 0.5
    batch_size = 1
    num_workers = 1
    device = 'cpu'
    image_path = "../../../../dev/shm/face_body_detection_and_association/Images/"
    matcher = HungarianMatcher()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']
    weights = "dummyheads.pth"
    model_type = "DDETR"

def boxes_dump(boxes, embs=None):
    if boxes.shape[-1] == 6:
        result = [{'box':[float(round(i, 1)) for i in box[:4].tolist()],
                    'score':float(round(float(box[5]), 5)),
                    'tag':int(box[4]) + 1} for box in boxes]
    else:
        result = [{'box':[float(round(i, 1)) for i in box[:4]],
                   'tag':int(box[4]) + 1} for box in boxes]
    return result

def save_json_lines(content,fpath):
    with open(fpath,'w') as fid:
        for db in content:
            line = json.dumps(db)+'\n'
            fid.write(line)
            
def export_ap_results(val_df,model,device= torch.device('cuda:0'), num= 100):
    valid_dataset = CrowdHuman_Dataset(
        train_df = val_df,
        transforms = get_valid_transforms()
    )   
    counter = 0
    all_results = []
    valid_data_loader = DataLoader(
                            valid_dataset,
                            batch_size = 1,#CFG.batch_size,
                            shuffle = False,
                            num_workers = CFG.num_workers,
                            collate_fn = collate_fn)
    for idx, batch in enumerate(tqdm(valid_data_loader)):
        images, masks =  batch["images"].to(device).decompose()
        targets = [{k: v for k, v in t.items()} for t in batch["labels"]]
        image_ids = batch["image_ids"]
        h = int(targets[0]["orig_size"][0])
        w = int(targets[0]["orig_size"][1])     
        model.eval()
        model.to(device)
        cpu_device = torch.device("cpu")    
        with torch.no_grad():
            outputs = model(pixel_values = images) 
        out_logits, out_bbox = outputs['logits'], outputs['pred_boxes']
        preds = []
        anns = []
        for i in range(len(targets)):
            prob = F.softmax(out_logits[i], -1)
            keep = prob.max(-1).values > 0.01        
            oboxes = out_bbox[i, keep]
            prob = prob[keep]
            sort = torch.argsort(prob[:, 0], descending = True)
            prob = prob[sort]
            oboxes = oboxes[sort]
            scores, labels = prob[..., :-1].max(-1)
            scores2 = []
            labels2 = []
            boxes = []
            for box, score, label in zip(oboxes, scores, labels):
                if label == 0:
                    boxes.append(box[:4])
                    boxes.append(box[4:])
                    scores2.append(score)
                    scores2.append(score)
                    labels2.append(torch.tensor([1]))
                    labels2.append(torch.tensor([0]))
            for box, score, label in zip(oboxes, scores, labels):
                if label == 1:
                    boxes.append(box[:4])
                    scores2.append(score)
                    labels2.append(torch.tensor([0]))
            gt_labels = []
            gt_bboxes = []
            bboxes = targets[i]['boxes']
            lab = targets[i]['labels']
            checked = False
            for box, t in zip(bboxes, lab):    
                if t == 0:
                    gt_bboxes.append(box[:4])
                    gt_bboxes.append(box[4:])
                    gt_labels.append(torch.tensor([1]))
                    gt_labels.append(torch.tensor([0]))                                         
                elif t == 1:
                    gt_bboxes.append(box[:4])
                    gt_labels.append(torch.tensor([0]))

            boxes = torch.stack([box for box in boxes])
            labels2 = torch.tensor(labels2).to(device)
            scores2 = torch.tensor(scores2).to(device)
            gt_bboxes = torch.stack([box for box in gt_bboxes])
            gt_labels = torch.tensor(gt_labels).to(device)
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            gt_bboxes = box_ops.box_cxcywh_to_xyxy(gt_bboxes)
            scale_fct = torch.tensor([w, h, w, h]).to(device)
            boxes = boxes.to(device) * scale_fct[None, :]
            gt_bboxes = gt_bboxes.to(device) * scale_fct[None, :]
            pred_boxes = torch.concat((boxes, labels2.unsqueeze(-1), scores2.unsqueeze(-1)), dim = -1).detach().cpu().numpy()
            gt_boxes = torch.concat((gt_bboxes, gt_labels.unsqueeze(-1)), dim = -1).detach().cpu().numpy()
            pred_boxes[:, 2:4] -= pred_boxes[:, :2]
            gt_boxes[:, 2:4] -= gt_boxes[:, :2]
            result_dict = dict(ID=image_ids[i] + ".jpg", height=int(h), width=int(w),
                dtboxes=boxes_dump(pred_boxes), gtboxes=boxes_dump(gt_boxes))
            all_results.append(result_dict)
        counter += 1
        if counter > num:
            break
    save_json_lines(all_results, "testing_dump_v2.json")
    
def export_mmr_res(val_df,model,img2id, device = torch.device('cuda:0'), num= 100):
    valid_dataset = CrowdHuman_Dataset(
        train_df = val_df,
        transforms = get_valid_transforms()
    )     
    all_results = []
    counter = 0
    valid_data_loader = DataLoader(
                            valid_dataset,
                            batch_size = 1,#CFG.batch_size,
                            shuffle = False,
                            num_workers = CFG.num_workers,
                            collate_fn = collate_fn)
    for idx, batch in enumerate(tqdm(valid_data_loader)):
        images, masks =  batch["images"].to(device).decompose()
        targets = [{k: v for k, v in t.items()} for t in batch["labels"]]
        image_ids = batch["image_ids"]
        h = int(targets[0]["orig_size"][0])
        w = int(targets[0]["orig_size"][1])     
        model.eval()
        model.to(device)
        cpu_device = torch.device("cpu")    
        with torch.no_grad():
            outputs = model(pixel_values = images) 
        out_logits, out_bbox = outputs['logits'], outputs['pred_boxes']
        preds = []
        anns = []
        for i in range(len(targets)):
            prob = F.softmax(out_logits[i], -1)
            keep = prob.max(-1).values > 0.1
            sort = torch.argsort(prob[keep][:, 0], descending = True)
            oboxes = out_bbox[i, keep]
            prob = prob[keep]
            prob = prob[sort]
            oboxes = oboxes[sort]
            scores, labels = prob[..., :-1].max(-1)
            scores2 = []
            labels2 = []
            boxes2 = []
            boxes = []
            scor = []
            for box, score, label in zip(oboxes, scores, labels):
                if label == 0:
                    boxes2.append(box[:4])
                    boxes2.append(box[4:])
                    scores2.append(score)
                    scores2.append(score)
                    labels2.append(torch.tensor([1]))
                    labels2.append(torch.tensor([0]))
            for box, score, label in zip(oboxes, scores, labels):
                if label == 1:
                    boxes.append(box[:4])
                    scor.append(score)
            boxes = torch.stack([box for box in boxes])
            if len(boxes2) > 0:
                boxes2 = torch.stack([box for box in boxes2])

            labels2 = torch.tensor(labels2).to(device)
            scores2 = torch.tensor(scores2).to(device)
            scor = torch.tensor(scor).to(device)
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            if len(boxes2) > 0:
                boxes2 = box_ops.box_cxcywh_to_xyxy(boxes2)
            scale_fct = torch.tensor([w, h, w, h]).to(device)

            boxes = boxes * scale_fct[None, :]
            if len(boxes2) > 0:
                boxes2 = boxes2 * scale_fct[None, :]
            boxes[:, 2:4] -= boxes[:, :2]
            if len(boxes2) > 0:
                boxes2[:, 2:4] -= boxes2[:, :2]
            boxes = boxes.detach().cpu().numpy()
            if len(boxes2) > 0:
                boxes2 = boxes2.detach().cpu().numpy()
            scores2 = scores2.detach().cpu().numpy()
            scor = scor.detach().cpu().numpy()
            if len(boxes2) > 0:
                for j in range(len(boxes2)):
                    if j%2 == 0:
                        content = {
                                'image_id': int(img2id[image_ids[i]]),
                                'category_id': 1,
                                'bbox':[float(round(k, 1)) for k in boxes2[j+1]],
                                'score':round(float(scores2[j+1]), 5),
                                'f_bbox':[float(round(k, 1)) for k in boxes2[j]],
                                'f_score':round(float(scores2[j]), 5)
                            }
                        all_results.append(content)
                    
            for j in range(len(boxes)):
                if j%2 == 0:
                    content = {
                            'image_id': int(img2id[image_ids[i]]),
                            'category_id': 1,
                            'bbox':[float(round(k, 1)) for k in boxes[j]],
                            'score':round(float(scor[j]), 5),
                            'f_bbox':[0.0, 0.0, 1.0, 1.0],
                            'f_score':float(round(0.0, 5))
                        }
                    all_results.append(content)         
            all_results.append(content)
        counter += 1
        if counter > num:
            break
    with(open("testing_res.json", "w")) as f:
        json.dump(all_results, f)
        
       

with(open("valid_df_body_face_head_v5.json", "r"))as f:
        val_df = json.load(f)
with(open("valid_df_body_face_head_ours2bfj.json", "r")) as f:
    ddd = json.load(f)
ddd = ddd["images"]
img2id = {}
id2img = {}
for i in ddd:
    img2id[i["file_name"].split('.')[0]] = i["id"]
    id2img[i["id"]] = i["file_name"].split('.')[0]
    
model = Detr_light(num_classes = CFG.num_classes, num_queries = CFG.num_queries)
checkpoint = torch.load(CFG.weights, map_location=CFG.device)
model.load_state_dict(checkpoint, strict=False)

export_mmr_res(val_df, model = model,img2id = img2id, num = 4370)
export_ap_results(val_df, model = model, num = 4370)
