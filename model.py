import os.path

import torch

import torch.nn as nn
import timm

from copy import deepcopy


def initialize_weights(model_name):
    # Initialize weights
    for m in model_name.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def freeze_layers(vit_model, freeze_blocks_ratio, freeze_patch_embed=True,
                  freeze_norm=True):
    layers = list(vit_model.blocks)
    num_layers_to_freeze = int(len(layers) * freeze_blocks_ratio)
    print(f'Freezing {num_layers_to_freeze} ({100 * freeze_blocks_ratio} %) ViT blocks')
    for layer in layers[:num_layers_to_freeze]:
        freeze_params(layer.parameters())
    if freeze_patch_embed:
        freeze_params(vit_model.patch_embed.parameters())
    if freeze_norm:
        freeze_params(vit_model.norm.parameters())


def create_model(model_name, freeze_ratio, ckpt_path=None):
    uni_model = timm.create_model(
        model_name, img_size=224, patch_size=16, num_classes=0, init_values=1e-5, dynamic_img_size=True
    )

    if ckpt_path is not None:
        uni_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")), strict=True)
        print(f'Loaded pretrained weights from {ckpt_path}')
    else:
        initialize_weights(uni_model)
        print('No pretrained weights loaded. Initialized weights.')
    if 1 >= freeze_ratio > 0:
        freeze_layers(uni_model, freeze_ratio)
    return uni_model


def get_trainable_params(_model):
    return sum(p.numel() for p in _model.parameters() if p.requires_grad)


class hiUNI(nn.Module):
    def __init__(self, n_classes, freeze_ratio, cmb, ckpt_path):
        super().__init__()

        backbone = create_model("vit_large_patch16_224", freeze_ratio, ckpt_path=ckpt_path)

        self.cmb = cmb
        self.cmb_len = len(cmb)

        self.backbones = nn.ModuleList()
        for _ in cmb:
            self.backbones.append(deepcopy(backbone))

        self.mlp = nn.Sequential(
            nn.Linear(self.cmb_len * 1024, 1024),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        outputs = [backbone(x[:, i, :, :, :]) for i, backbone in enumerate(self.backbones)]
        x = torch.stack(outputs, dim=1)  # [batch_size, cmb_len, 1024]
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


# for debugging
if __name__ == '__main__':
    combination = 'sml'
    hf_weight_path = r'pytorch_model.bin'  # replace with yours
    assert os.path.exists(hf_weight_path), "Please download the weights from HuggingFace and replace the path."
    model = hiUNI(n_classes=4, freeze_ratio=0.6, cmb=combination,
                  ckpt_path=hf_weight_path).cuda()  # from HuggingFace
    # print(model)
    x = torch.randn(2, 3, 3, 224, 224).cuda()  # [batch_size, cmb_len, 3, 224, 224]
    output = model(x)
    print(output.shape)
    param_number = get_trainable_params(model)
    print(f'Trainable parameters: {round(param_number / 1e6, 2)}M')

