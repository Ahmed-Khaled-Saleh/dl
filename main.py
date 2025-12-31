from fastcore import *
from fastcore.utils import *
import torch
import argparse

import time
import os
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import ConcatDataset
from omegaconf import OmegaConf
from dotenv import load_dotenv
import torch.nn as nn

import torch.optim as optim

from dl.data import init_data
from dl.model import init_model
from dl.wandb_writer import WandbWriter
from dl.trainer import Trainer



def main(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = init_data(cfg)
    model = init_model(cfg)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr)
    writer = WandbWriter(cfg)

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    trainer = Trainer(cfg, model, loaders, criterion=criterion, 
                      optimizer=optimizer, device=device, writer=writer)

    model, optimizer, train_losses, val_losses = None, None, None, None
    if cfg.task in ['probing', "fine-tuning"]:
        model, optimizer, train_losses, val_losses = trainer.fit()

    df_result = trainer.predict()
    return model, optimizer, train_losses, val_losses, df_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DL Training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--timestamp', type=str, help='Time stamp', required=True)
    parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)

    args = parser.parse_args()

    
    if args.env_file:
        load_dotenv(args.env_file)
        key = os.getenv("WANDB_API_KEY", None)
        if key:
            os.environ["WANDB_API_KEY"] = key

    try:
        cfg = OmegaConf.load(args.config)
    except:
        print("Invalid config file path")

    cfg.now = args.timestamp 

    model, optimizer, train_losses, val_losses, df_result = main(cfg)

    save_path = os.path.join(cfg.root_dir, cfg.log_dir, cfg.project_name, 
                             cfg.model.name, cfg.task)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ckpt_path = os.path.join(save_path, "ckpt_final.pt")
        
    if model:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        
        torch.save(ckpt, ckpt_path)

    result_path = os.path.join(save_path, "results.csv")
    df_result.to_csv(result_path, index=False)