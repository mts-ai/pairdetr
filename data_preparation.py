import json
import os 
import cv2
from typing import Dict, List, Any
from config import CFG
    
def get_data_annotations():    
    if os.path.isfile(CFG.train_annotation_path):
        with(open(CFG.train_annotation_path, "r"))as f:
            train_df = json.load(f)
    else:
        print("NOT ABLE TO FIND THE ANNOTATION FILE FOR TRAINING ")
    if os.path.isfile(CFG.val_annotation_path):
        with(open(CFG.val_annotation_path, "r"))as f:
            val_df = json.load(f)
    else:
        print("NOT ABLE TO FIND THE ANNOTATION FILE FOR VALIDATION ")
    return train_df, val_df

