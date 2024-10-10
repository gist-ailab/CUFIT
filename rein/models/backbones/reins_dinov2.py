from .reins import Reins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train


class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )

        # self.reins2 = Reins(
        #     num_layers = kwargs['depth'],
        #     embed_dims = kwargs['embed_dim'],
        #     patch_size = kwargs['patch_size'],
        # )

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x

    def forward_features_full_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)



    def forward_features_no_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "linear"])
        set_train(self, ["reins", "linear"])



class ReinsDinoVisionTransformer_3_head(DinoVisionTransformer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins1 = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )

        self.reins2 = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )

    def forward_features1(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins1.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x
    
    def forward_features2(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x

    def forward_features_no_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins1", "reins2", "linear"])
        set_train(self, ["reins1", "reins2", "linear"])