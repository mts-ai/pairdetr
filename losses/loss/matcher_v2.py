### this file is copied from the original DETR repo with some changes 
# Mainly we changed the cost calculation to take pairs into account using head or face according to what available
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ortools.graph.python import min_cost_flow

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 2, cost_bbox: float = 5, cost_giou: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    # if we want to use IB cost on matching also   
    def get_IB(self, target_boxes, src_boxes):
        imaginary_gt_boxes = []
        for box in target_boxes:
            x = min(box[0], box[4])
            y = min(box[1], box[5])
            w = max(box[0], box[4]) - x
            h = max(box[1], box[5]) - y
            cx = x + w/2
            cy = y + h/2
            imaginary_gt_boxes.append([cx,cy,w,h])
        imaginary_gt_boxes = torch.tensor(imaginary_gt_boxes)
        imaginary_pred_boxes = []
        for box in src_boxes:
            x = min(box[0], box[4])
            y = min(box[1], box[5])
            w = max(box[0], box[4]) - x
            h = max(box[1], box[5]) - y
            cx = x + w/2
            cy = y + h/2
            imaginary_pred_boxes.append([cx,cy,w,h])
        imaginary_pred_boxes = torch.tensor(imaginary_pred_boxes)
        return imaginary_gt_boxes, imaginary_pred_boxes
    @torch.no_grad()
    def forward(self, outputs, targets, model_type):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if model_type == "DETR":
            key_logits = "pred_logits"
        else:
            key_logits = "logits"
        bs, num_queries = outputs[key_logits].shape[:2]
        out_prob = outputs[key_logits].flatten(0, 1).softmax(-1) 
        out_bbox = outputs["pred_boxes"].flatten(0, 1)     
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) 
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        # split pairs to boxes (body, face) or (body, head)
        # or to (body, face) or (body, auto-generated) depends on which annotation file you are using.
        out_bbox1 = out_bbox[:,:4]
        out_bbox2 = out_bbox[:,4:]
        tgt_bbox1 = tgt_bbox[:,:4]
        tgt_bbox2 = tgt_bbox[:,4:]
        cost_bbox1 = torch.cdist(out_bbox1, tgt_bbox1, p=1)
        cost_bbox2 = torch.cdist(out_bbox2, tgt_bbox2, p=1)    
        cost_giou1 =  1 - generalized_box_iou(box_cxcywh_to_xyxy(out_bbox1), box_cxcywh_to_xyxy(tgt_bbox1))
        cost_giou2 =  1 - generalized_box_iou(box_cxcywh_to_xyxy(out_bbox2), box_cxcywh_to_xyxy(tgt_bbox2))
        C1 = self.cost_bbox * cost_bbox1 + self.cost_class * cost_class + self.cost_giou * cost_giou1 
        C2 = self.cost_bbox * cost_bbox2 + self.cost_class * cost_class + self.cost_giou * cost_giou2
        C = C1.cpu() +C2.cpu() 
        C = C.view(bs, num_queries , -1)
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
