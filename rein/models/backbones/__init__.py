# from .dino_v2 import DinoVisionTransformer
from .reins_dinov2 import ReinsDinoVisionTransformer, ReinsDinoVisionTransformer_3_head
from .reins_resnet import ReinsResNet
# from .reins_eva_02 import ReinsEVA2
# from .clip import CLIPVisionTransformer

__all__ = [
    "CLIPVisionTransformer",
    "DinoVisionTransformer",
    "ReinsDinoVisionTransformer",
    "ReinsDinoVisionTransformer_3_head",
    "ReinsEVA2",
]
