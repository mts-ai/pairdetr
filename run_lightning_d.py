import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from dataset import *
from config import CFG
from train_lightning_d import *
from data_preparation import get_data_annotations
from pytorch_lightning import Trainer
from tqdm.notebook import tqdm
from transformers import AutoImageProcessor
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from zipfile import ZipFile
import torch.distributed as dist
from typing import Optional, List
from torch import Tensor
from utils import collate_fn 

def run(train_df, val_df):
    # seed to ensure reproducability
    seed_everything(CFG.seed, workers=True)  
    # loading datasets               
    train_dataset = CrowdHuman_Dataset(
        train_df = train_df,
        transforms = get_train_transforms()
    )
    valid_dataset = CrowdHuman_Dataset(
        train_df = val_df,
        transforms = get_valid_transforms()
    )
    # preparing dataloaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size = CFG.batch_size,
        shuffle = False,
        num_workers = CFG.num_workers,
        collate_fn = collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size = CFG.batch_size,
        shuffle = False,
        num_workers = CFG.num_workers,
        collate_fn = collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    # Initialize the model 
    model = Detr_light(num_classes = CFG.num_classes,num_queries = CFG.num_queries)
    # load the checkpoint if any (if trained stage1 without association for example)
    if CFG.checkpoint_id is not None:
        state_dict = torch.load(weights_path)
        model_state_dict = model.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            state_dict.pop("optimizer_states", None)
        model.load_state_dict(state_dict, strict=False)
    # setting up pytorch lightning trainer
    checkpoint_callback = ModelCheckpoint(**CFG.checkpoint["checkpoint_details"])
    trainer = Trainer(strategy="ddp_find_unused_parameters_false", accelerator='gpu', max_epochs=CFG.epochs, gradient_clip_val = CFG.gradient_clip_val, enable_checkpointing = True, replace_sampler_ddp=True,deterministic="warn", callbacks = [checkpoint_callback])
    trainer.fit(model, train_data_loader, valid_data_loader)
    torch.save(model.state_dict(), CFG.save_weights_as)
    task.close()
def main():
    train_df, val_df = get_data_annotations()
    run(train_df, val_df)
if __name__ == "__main__":
    main()

