
_small_variant = dict(
    patch_size=14,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    img_size=518,
    ffn_layer="mlp",
    init_values=1e-05,
    block_chunks=0,
    qkv_bias=True,
    proj_bias=True,
    ffn_bias=True
)
_small_dino = 'dinov2_vits14'

_base_variant = dict(
    patch_size=14,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    img_size=518,
    ffn_layer="mlp",
    init_values=1e-05,
    block_chunks=0,
    qkv_bias=True,
    proj_bias=True,
    ffn_bias=True,
    out_indices = [7, 11, 14, 17]
)
_base_dino = 'dinov2_vitb14'

_large_variant = dict(
    patch_size=14,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    img_size=518,
    ffn_layer="mlp",
    init_values=1e-05,
    block_chunks=0,
    qkv_bias=True,
    proj_bias=True,
    ffn_bias=True
)
_large_dino = 'dinov2_vitl14'
