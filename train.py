import argparse
import os
import torch
import torch.nn as nn
import yaml

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PathologyDatasetKFold
from utils.metrics import val_auc
from model import hiUNI


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Train hi-UNI")
    parser.add_argument('--fold', type=int, default=1, help='Fold for cross-validation')
    args = parser.parse_args()
    config_keys = ['batch_size', 'lr', 'freeze_ratio', 'cmb',
                   'epochs', 'iters_to_val', 'save_best', 'UNI_path',
                   'class_names']
    for key in config_keys:
        setattr(args, key, load_config()[key])
    return args


def print_configs(_args):
    print('============== Configs ==============')
    for key, value in vars(_args).items():
        print(f'{key:<10}  \t {value}', end='\n')
    print('=====================================')


if __name__ == '__main__':
    args = parse_args()

    combination = args.cmb

    print_configs(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = args.class_names
    num_classes = len(class_names)

    # kf dir check
    assert os.path.exists(f'./kf'), 'k-fold dir not exists, run utils/gen_kfold_split.py first'

    # create dataset objects for train and val
    train_dataset = PathologyDatasetKFold(mode='train', combination=combination, fold=args.fold)
    val_dataset = PathologyDatasetKFold(mode='val', combination=combination, fold=args.fold)

    # create data loaders for train and val
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    model = hiUNI(n_classes=num_classes, freeze_ratio=args.freeze_ratio, cmb=args.cmb, ckpt_path=args.UNI_path)
    model.to(device)

    criterion_train = nn.CrossEntropyLoss().to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)

    # scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    store_path = f'runs/{combination}_{args.freeze_ratio}/{args.fold}/'
    os.makedirs(store_path, exist_ok=True)

    best_slide_auc = 0

    for epoch in range(args.epochs):
        epoch_avg_loss = 0

        print(f'Current epoch : {epoch + 1}, LR : {optimizer.param_groups[0]["lr"]:.8f}')

        for iter_idx, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            data = data.to(device)
            label = label.to(device)  # [batch_size, num_classes]

            output = model(data)  # [batch_size, num_classes]

            loss = criterion_train(output, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if ((args.iters_to_val != -1 and iter_idx % args.iters_to_val == 0 and iter_idx != 0)
                    or (args.iters_to_val == -1 and iter_idx == len(train_loader) - 1)):
                model.eval()
                print(f"Epoch {epoch + 1} Iter {iter_idx}\n")

                slide_auc = val_auc(model=model, loader=val_loader, criterion=criterion_val,
                                    class_names=class_names, save_path=store_path,
                                    iter_idx=iter_idx)
                model_path = f'{args.fold}_best.pth'

                if args.save_best is True and slide_auc > best_slide_auc:
                    best_slide_auc = slide_auc
                    torch.save(model, os.path.join(store_path, model_path))
                    print(f'Saving best model to {model_path}, current best auc: {best_slide_auc:.4f}')
        scheduler.step()
