import os 
import numpy as np
import torch
import albumentations as A
import random
from config import CFG
from dataset import *
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from torch import Tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = True
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def collate_fn(batch):
    ## Notice there will be difference if used original DETR preprocessing or HF processor what is important is to keep the same method during inference
    pixel_values = [item[0] for item in batch]
    pixel_values = nested_tensor_from_tensor_list(pixel_values)
    #encoding = image_processor(images=pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    batch = {}
    batch['images'] = pixel_values
    batch['labels'] = labels
    batch["image_ids"] = ids
    return batch

class BaseMetricResults(dict):
    """Base metric class, that allows fields for pre-defined metrics."""

    def __getattr__(self, key: str) -> Tensor:
        # Using this you get the correct error message, an AttributeError instead of a KeyError
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: Tensor) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self:
            del self[key]
        raise AttributeError(f"No such attribute: {key}")
        
class COCOMetricResults(BaseMetricResults):
    """Class to wrap the final COCO metric results including various mAP/mAR values."""

    __slots__ = (
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
        "map_per_class",
        "mar_100_per_class",
    )
    
def compute(mAP) -> dict:
    """Computes metric."""
    classes = mAP._get_classes()
    precisions, recalls = mAP._calculate(classes)
    map_val, mar_val = mAP._summarize_results(precisions, recalls)

    # if class mode is enabled, evaluate metrics per class
    map_per_class_values: Tensor = torch.tensor([-1.0])
    mar_max_dets_per_class_values: Tensor = torch.tensor([-1.0])
    if mAP.class_metrics:
        map_per_class_list = []
        mar_max_dets_per_class_list = []

        for class_idx, _ in enumerate(classes):
            cls_precisions = precisions[:, :, class_idx].unsqueeze(dim=2)
            cls_recalls = recalls[:, class_idx].unsqueeze(dim=1)
            cls_map, cls_mar = mAP._summarize_results(cls_precisions, cls_recalls)
            map_per_class_list.append(cls_map.map_50)
            mar_max_dets_per_class_list.append(cls_mar[f"mar_{mAP.max_detection_thresholds[-1]}"])

        map_per_class_values = torch.tensor(map_per_class_list, dtype=torch.float)
        mar_max_dets_per_class_values = torch.tensor(mar_max_dets_per_class_list, dtype=torch.float)

    metrics = COCOMetricResults()
    metrics.update(map_val)
    metrics.update(mar_val)
    metrics.map_per_class = map_per_class_values
    metrics[f"mar_{mAP.max_detection_thresholds[-1]}_per_class"] = mar_max_dets_per_class_values
    return metrics