from .reins import Reins
from .utils import set_requires_grad, set_train
from typing import List, Dict
import torch.nn as nn
import timm
from timm.models.resnet import ResNet, Bottleneck

# Modified from the code of https://github.com/w1oves/Rein/blob/train/rein/models/backbones/reins_resnet.py
class ReinsResNet(ResNet):
    def __init__(
        self,
        **kwargs,
    ):
        model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
        super().__init__(**dict(model_args, **kwargs))
        self.reins: List[Reins] = nn.ModuleList()
        self.reins.append(Reins(num_layers=1, embed_dims=256, patch_size=1)) # For layer 1
        self.reins.append(Reins(num_layers=1, embed_dims=512, patch_size=1)) # For layer 1
        self.reins.append(Reins(num_layers=1, embed_dims=1024, patch_size=1)) # For layer 1
        self.reins.append(Reins(num_layers=1, embed_dims=2048, patch_size=1)) # For layer 1


        print('length of reins: ', len(self.reins))
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            res_layer = getattr(self, layer_name)
            # print(res_layer)
            x = res_layer(x)
            # print(x.shape)
            B, C, H, W = x.shape
            x = (
                self.reins[i]
                .forward(
                    x.flatten(-2, -1).permute(0, 2, 1),
                    0,
                    batch_first=True,
                    has_cls_token=False,
                )
                .permute(0, 2, 1)
                .reshape(B, C, H, W)
            )
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "fc"])
        set_train(self, ["reins", "fc"])
