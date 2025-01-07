import torch

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from PIL import Image


transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class PathologyDataset(Dataset):
    def __init__(self, mode, combination):
        self.mode = mode
        self.combination = combination
        self.data = pd.read_csv(f'{mode}.csv')

        self.name = self.data['name']
        self.tile_paths = {
            's': self.data['tile-s'],
            'm': self.data['tile-m'],
            'l': self.data['tile-l']
        }
        self.y_data = self.data['label']
        self.transform = transform

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # load tile
        name = self.name.iloc[index]
        tiles = []

        for combination_key in self.combination:
            tile_path = self.tile_paths[combination_key].iloc[index]
            tile = self.transform(Image.open(tile_path))
            tiles.append(tile)

        tile = torch.stack(tiles)

        # load label
        label = self.y_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)

        if self.mode == 'train':
            return tile, label
        elif self.mode == 'val':
            return name, tile, label
        return None


class PathologyDatasetKFold(Dataset):
    def __init__(self,  mode, combination, fold):
        self.mode = mode
        self.combination = combination
        self.data = pd.read_csv(f'kf/{fold}_{mode}.csv')

        self.name = self.data['name']
        self.tile_paths = {
            's': self.data['tile-s'] if 'tile-s' in self.data.columns else None,
            'm': self.data['tile-m'] if 'tile-m' in self.data.columns else None,
            'l': self.data['tile-l'] if 'tile-l' in self.data.columns else None
        }
        self.y_data = self.data['label']
        self.transform = transform

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # load tile
        name = self.name.iloc[index]
        tiles = []

        for combination_key in self.combination:
            tile_path = self.tile_paths[combination_key].iloc[index]
            tile = self.transform(Image.open(tile_path))
            tiles.append(tile)

        tile = torch.stack(tiles)

        # load label
        label = self.y_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)

        if self.mode == 'train':
            return tile, label
        elif self.mode == 'val':
            return name, tile, label
        return None
