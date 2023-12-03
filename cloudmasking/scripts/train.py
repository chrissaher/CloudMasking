import os
import argparse
import copy
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from cloudmasking.models.registry import get_model
from cloudmasking.dataset.clouddataset import CloudDataset


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

def train_loop(args):

    output_dir = Path(args.output_dir) / args.exp_id
    if output_dir.exists():
        print(f"The experiment {args.exp_id} already exists. Chose a different experiment id.")
        exit(0)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    threshold = args.threshold
    criterion = args.criterion
    patience = args.patience
    random_resize_crop = None
    if args.random_resize_crop is not None:
        random_resize_crop = (args.random_resize_crop, args.random_resize_crop)
    disable_wandb = args.disable_wandb


    model = get_model(args.model)

    train_dataset = CloudDataset(Path(args.train_dir), random_resize_crop=random_resize_crop)
    val_dataset = CloudDataset(Path(args.val_dir), random_resize_crop=random_resize_crop)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss() if args.criterion == "bce" else FocalLoss()

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
    scheduler = None
    if patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)

    model = model.to(device)

    best_accuracy = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    binary_acc_metric = tm.classification.BinaryAccuracy(threshold=threshold).to(device)
    precision = tm.classification.BinaryPrecision(threshold=threshold).to(device)
    recall = tm.classification.BinaryRecall(threshold=threshold).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        _ = model.train()
        train_losses = list()
        for index, batch in tqdm(enumerate(train_dataloader)):
            input_data, mask = batch
            input_data = input_data.to(device)
            mask = mask.to(device)
            output = model(input_data)
            loss = criterion(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if not disable_wandb:
                wandb.log({'train/loss': loss.cpu().item()})

        _ = model.eval()
        valid_losses = list()
        all_preds = list()
        all_true = list()
        for index, batch in tqdm(enumerate(valid_dataloader)):
            input_data, mask = batch
            input_data = input_data.to(device)
            mask = mask.to(device)
            output = model(input_data)
            valid_losses.append(loss.cpu().item())

            # For torch metric
            bacc = binary_acc_metric(output.reshape(-1), mask.reshape(-1))
            pre = precision(output.reshape(-1), mask.reshape(-1))
            rec = recall(output.reshape(-1), mask.reshape(-1))


        # Calculate metrics
        acc = binary_acc_metric.compute().cpu().item()
        pre = precision.compute().cpu().item()
        rec = recall.compute().cpu().item()
        f1 = 2 * pre * rec / (pre + rec)

        binary_acc_metric.reset()
        precision.reset()
        recall.reset()

        # Log to wandb
        if not disable_wandb:
            wandb.log({'valid/loss': np.average(valid_losses)})
            wandb.log({'valid/accuracy': acc})
            wandb.log({'valid/precision': pre})
            wandb.log({'valid/recall': rec})
            wandb.log({'valid/f1': f1})

            wandb.log({'train/lr': optimizer.param_groups[0]['lr']})

        if scheduler is not None:
            scheduler.step(np.mean(train_losses))

        if acc > best_accuracy:
             best_accuracy = acc
             best_model_wts = copy.deepcopy(model.state_dict())
             torch.save(model.state_dict(), checkpoint_dir / f"best_model_{epoch + 1}.pth")

    torch.save(model.state_dict(), checkpoint_dir / f"best_model_final.pth")

def parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='Training procedure')
    # parser.add_argument('--task', type=str, default='BEFORE', help='task to run (BEFORE or BEFOREAFTER)')
    # parser.add_argument('--train', type=str, default='data/lettercounting-train.txt', help='path to train examples')
    # parser.add_argument('--dev', type=str, default='data/lettercounting-dev.txt', help='path to dev examples')
    parser.add_argument('--exp_id', type=str, required=True, help='Id of the experiment. Must be unique')
    parser.add_argument('-od', '--output_dir', type=str, default='./output', help='path to store the output of training procedure.')

    # Optimization args
    parser.add_argument('-c','--criterion', type=str, default='bce', help='Loss function. Supported values are `bce`, `focal`.')

    # Data args
    parser.add_argument('-td', '--train_dir', type=str, help='Directory with training images and masks.')
    parser.add_argument('-vd', '--val_dir', type=str, help='Directory with valdation images and masks.')

    # Model args
    parser.add_argument('-m', '--model', type=str, default='unet', help='Model to use. Supported values are (`unet`, `segformer`, `deeplabv3`).')
    parser.add_argument('-th','--threshold', type=float, default=0.5, help='Threhold for the probability to become of certain class. If prob > 0.5, then class = 1.')

    # Augmentation args
    parser.add_argument('-rrc','--random_resize_crop', type=int, default=None, help='Random resize crop size. Expected an int, and it will be duplicated to keep square shape.')

    # Learning args
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Training batch size; 1 by default and you do not need to batch unless you want to.')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('-p', '--patience', type=int, default=0, help='patience of the scheduler')

    # Wandb args
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging.')

    args = parser.parse_args()
    return args


def get_env(env_var):
    if os.environ.get(env_var):
        return os.environ.get(env_var)
    else:
        raise Exception(str(env_var) , ' is necessary to enable wandb')

def wandb_setup(args):
    api_key = get_env("WANDB_API_KEY")
    project = "cloud-segmentation" # Fixed for this project

    wandb.login(key=api_key)
    kwargs = {
        "project": project,
        "name": args.exp_id,
        "config": vars(args),
        "allow_val_change": False,
        "resume": "never",
    }
    wandb.init(**kwargs)


def main():

    args = parse_args()
    print(args)

    if not args.disable_wandb:
        wandb_setup(args)

    train_loop(args)


if __name__ == '__main__':
    main()
