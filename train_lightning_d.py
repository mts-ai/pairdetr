import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from transformers import DeformableDetrForObjectDetection, DetaForObjectDetection
from utils import AverageMeter, BaseMetricResults, COCOMetricResults, compute 
from config import CFG
from tqdm.autonotebook import tqdm
import sys 
sys.path.append('./losses/')
from losses.loss.detr_v2 import SetCriterion
from losses.util import box_ops
from typing import Any, Dict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import math
from torch import Tensor

# helper function from hugginface 
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
# helper function from hugginface 
class DeformableDetrMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers:int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
# Model class
class Detr_light(pl.LightningModule):
    def __init__(self, num_classes: int, num_queries: int) -> None:
        super(Detr_light, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries        
        self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")        
        if CFG.freeze_backbone:
            for param in self.model.model.backbone.parameters():
                param.requires_grad = False 
        self.in_features = self.model.class_embed[0].in_features        
        self.model.model.query_position_embeddings = nn.Embedding(self.num_queries, 512)
        self.class_embed = nn.Linear(self.in_features, self.num_classes)
        self.bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=256, hidden_dim=256, output_dim=8, num_layers=3
        )
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        # replace the output layers of DETR with different ones to filt pair detection 
        self.model.class_embed = nn.ModuleList([self.class_embed for _ in range(6)])
        self.model.bbox_embed  = nn.ModuleList([self.bbox_embed for _ in range(6)])
        
        self.criterion = SetCriterion(CFG.num_classes-1, CFG.matcher, CFG.weight_dict, eos_coef = CFG.null_class_coef, losses = CFG.losses, model_type = CFG.model_type)
        self.train_loss = AverageMeter()
        self.valid_loss = AverageMeter()
        # there is difference between torchmetrics AP results and COCO one the results reported on the paper are using validation scripts these AP results will be higher.
        self.mAP = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics = True, max_detection_thresholds = [1000] )
        self.postprocess = PostProcess()
    # Custom forward to use the relative point adaptively
    def forward(self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,) -> torch.Tensor:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        outputs = self.model.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]
        outputs_classes = []
        outputs_coords = []
        cons = inverse_sigmoid(init_reference)
        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.model.class_embed[level](hidden_states[:, level])
            delta_bbox = self.model.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                delta_bbox[..., :4] += reference
                outputs_coord_logits = delta_bbox
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                delta_bbox[..., 4:6] += cons
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes, dim=1)
        outputs_coord = torch.stack(outputs_coords, dim=1)

        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        dict_outputs = {
            "logits":logits,
            "pred_boxes": pred_boxes,
            "init_reference_points": outputs.init_reference_points,
                       }    
        return dict_outputs

    def training_step(self, batch, batch_idx) -> AverageMeter:
        self.model.train()
        self.criterion.train()      
        images, masks =  batch["images"].to(self.device).decompose()
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        output = self(pixel_values = images)
        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)                 
        self.train_loss.update(losses.item(), CFG.batch_size)        
        return losses

    def validation_step(self,  batch, batch_idx) -> AverageMeter:
        self.model.eval()
        self.criterion.eval() 
        ans = []
        preds= []
        with torch.no_grad():            
            images, masks =  batch["images"].to(self.device).decompose()
            image_ids = batch["image_ids"]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
            output = self(pixel_values = images)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict        
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)            
            self.valid_loss.update(losses.item(), CFG.batch_size)                
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            preds, anns = self.postprocess(output, targets, orig_target_sizes, self.device)
            self.mAP.update(preds, anns)              
        return losses
    
    def configure_optimizers(self): 
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if ("backbone" not in n) and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if ("backbone" in n) and p.requires_grad],
                  "lr": CFG.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=CFG.lr, weight_decay = CFG.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.drop_lr_at_epoch, gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
            
    def on_validation_epoch_end(self) -> None:
        mAPs = {"val_" + k: v for k, v in compute(self.mAP).items()}
        map_at_50 = mAPs.pop("val_map_50")
        map_per_class = mAPs.pop("val_map_per_class")
        map_small = mAPs.pop("val_map_small")
        map_medium = mAPs.pop("val_map_medium")
        map_large = mAPs.pop("val_map_large")
        mar_per_class = mAPs.pop("val_mar_1000_per_class")
        self.log("mAP", map_at_50)
        self.mAP.reset()


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, targets, target_sizes, device):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        preds = []
        anns = []
        for i in range(len(targets)):
            prob = F.softmax(out_logits[i], -1)
            keep = prob.max(-1).values > CFG.conf_threshold
            sort = torch.argsort(prob[keep][:, 0], descending = True)
            oboxes = out_bbox[i, keep]
            prob = prob[keep]
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

            labels2 = torch.tensor(labels2)
            scores2 = torch.tensor(scores2)
            gt_bboxes = torch.stack([box for box in gt_bboxes])
            gt_labels = torch.tensor(gt_labels)

            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            gt_bboxes = box_ops.box_cxcywh_to_xyxy(gt_bboxes)
            img_h, img_w = target_sizes[i]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(device)

            boxes = boxes * scale_fct[None, :]
            gt_bboxes = gt_bboxes * scale_fct[None, :]

            preds.append(dict(
                boxes = boxes.to(device),
                scores = scores2.to(device),
                labels = labels2.to(device)                
            ))
            anns.append(dict(
                boxes = gt_bboxes.to(device),
                labels = gt_labels.to(device)                
            ))
        return preds, anns
