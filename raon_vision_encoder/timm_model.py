# Originally from OpenCLIP (https://github.com/mlfoundations/open_clip)

import logging
import types
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import timm
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d
    from timm.layers import Mlp, to_2tuple
    from timm.layers import AttentionRope, RotaryEmbeddingCat
except ImportError:
    timm = None


class TimmModel(nn.Module):
    """timm model adapter"""

    def __init__(
        self,
        model_name: str,
        embed_dim: int,
        image_size: Union[int, Tuple[int, int]] = 224,
        pool: str = "avg",
        proj: str = "linear",
        proj_bias: bool = False,
        drop: float = 0.0,
        drop_path: Optional[float] = None,
        patch_drop: Optional[float] = None,
        init_values: Optional[float] = None,
        qk_norm: bool = False,
        use_rope: bool = False,
        rope_keep_ape: bool = False,
        dynamic_img_size: bool = False,
        norm_pre: bool = False,
        pretrained: bool = False,
        output_tokens: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError(
                "Please install the latest timm (`pip install timm`) to use timm based models."
            )
        self.image_size = to_2tuple(image_size)
        self.output_tokens = output_tokens

        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs["drop_path_rate"] = drop_path
        if patch_drop is not None:
            timm_kwargs["patch_drop_rate"] = patch_drop
        if init_values is not None:
            timm_kwargs["init_values"] = init_values
        if qk_norm:
            timm_kwargs["qk_norm"] = True
        if dynamic_img_size:
            timm_kwargs["dynamic_img_size"] = True
        if use_rope:

            class _AttentionRopeNoPrefix(AttentionRope):
                """AttentionRope with num_prefix_tokens=0 for models without cls token."""

                def __init__(self, *args, **kwargs):
                    kwargs["num_prefix_tokens"] = 0
                    super().__init__(*args, **kwargs)

            timm_kwargs["attn_layer"] = _AttentionRopeNoPrefix
            if not rope_keep_ape:
                timm_kwargs["pos_embed"] = "none"

        custom_pool = pool in ("abs_attn", "rot_attn")
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            proj_dim = 0 if proj == "none" else embed_dim
            self.trunk = timm.create_model(
                model_name,
                num_classes=proj_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get("pool_size", None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                self.trunk.reset_classifier(0, global_pool="")
            else:
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        if pool == "abs_attn":
            head_layers["pool"] = AbsAttentionPool2d(
                prev_chs, feat_size=feat_size, out_features=embed_dim
            )
            prev_chs = embed_dim
        elif pool == "rot_attn":
            head_layers["pool"] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        if proj == "linear":
            head_layers["drop"] = nn.Dropout(drop)
            head_layers["proj"] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(
                prev_chs,
                2 * embed_dim,
                embed_dim,
                drop=(drop, 0),
                bias=(True, proj_bias),
            )

        self.head = nn.Sequential(head_layers)

        if (
            norm_pre
            and hasattr(self.trunk, "norm_pre")
            and isinstance(self.trunk.norm_pre, nn.Identity)
        ):
            self.trunk.norm_pre = nn.LayerNorm(self.trunk.embed_dim)
            logging.info(
                f"Replaced norm_pre Identity with LayerNorm({self.trunk.embed_dim})"
            )

        self._has_rope = use_rope
        if use_rope:
            self._setup_rope()

    def _setup_rope(self):
        """Inject 2D Rotary Position Embedding into the timm trunk."""
        num_heads = self.trunk.blocks[0].attn.num_heads
        head_dim = self.trunk.embed_dim // num_heads

        self.trunk.patch_embed.strict_img_size = False

        self.rope = RotaryEmbeddingCat(
            dim=head_dim,
            max_res=max(self.image_size),
            in_pixels=True,
        )

        def _block_forward_rope(block_self, x, rope=None, attn_mask=None):
            x = x + block_self.drop_path1(
                block_self.ls1(
                    block_self.attn(block_self.norm1(x), rope=rope, attn_mask=attn_mask)
                )
            )
            x = x + block_self.drop_path2(
                block_self.ls2(block_self.mlp(block_self.norm2(x)))
            )
            return x

        for blk in self.trunk.blocks:
            blk.forward = types.MethodType(_block_forward_rope, blk)

        timm_model_ref = self
        _num_prefix = getattr(self.trunk, "num_prefix_tokens", 0)

        def _forward_features_rope(trunk_self, x, attn_mask=None):
            from torch.utils.checkpoint import checkpoint
            from timm.layers import resample_abs_pos_embed

            ps = trunk_self.patch_embed.patch_size
            grid_shape = [x.shape[2] // ps[0], x.shape[3] // ps[1]]

            x = trunk_self.patch_embed(x)
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1, x.shape[-1])
            if hasattr(trunk_self, "pos_embed") and trunk_self.pos_embed is not None:
                if x.shape[1] != trunk_self.pos_embed.shape[1]:
                    x = x + resample_abs_pos_embed(
                        trunk_self.pos_embed, grid_shape, num_prefix_tokens=_num_prefix
                    )
                else:
                    x = x + trunk_self.pos_embed
            x = trunk_self.pos_drop(x)
            x = trunk_self.norm_pre(x)

            rot_pos_embed = timm_model_ref.rope.get_embed(shape=grid_shape)

            _sdpa_mask = None
            if attn_mask is not None:
                _sdpa_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
                _sdpa_mask.masked_fill_(~attn_mask, float("-inf"))
                _sdpa_mask = _sdpa_mask.unsqueeze(1).unsqueeze(2)

            for blk in trunk_self.blocks:
                if trunk_self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(
                        blk,
                        x,
                        rope=rot_pos_embed,
                        attn_mask=_sdpa_mask,
                        use_reentrant=False,
                    )
                else:
                    x = blk(x, rope=rot_pos_embed, attn_mask=_sdpa_mask)

            x = trunk_self.norm(x)
            return x

        self.trunk.forward_features = types.MethodType(
            _forward_features_rope, self.trunk
        )

    def _setup_dynamic_pos_embed(self):
        """Patch forward_features for variable-resolution pos_embed interpolation (non-RoPE)."""
        self.trunk.patch_embed.strict_img_size = False
        _num_prefix = getattr(self.trunk, "num_prefix_tokens", 0)

        def _forward_features_dynamic(trunk_self, x, patch_valid_mask=None):
            from torch.utils.checkpoint import checkpoint
            from timm.layers import resample_abs_pos_embed

            ps = trunk_self.patch_embed.patch_size
            grid_shape = [x.shape[2] // ps[0], x.shape[3] // ps[1]]

            x = trunk_self.patch_embed(x)
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1, x.shape[-1])
            if hasattr(trunk_self, "pos_embed") and trunk_self.pos_embed is not None:
                if x.shape[1] != trunk_self.pos_embed.shape[1]:
                    x = x + resample_abs_pos_embed(
                        trunk_self.pos_embed, grid_shape, num_prefix_tokens=_num_prefix
                    )
                else:
                    x = x + trunk_self.pos_embed
            x = trunk_self.pos_drop(x)
            x = trunk_self.norm_pre(x)

            _sdpa_mask = None
            if patch_valid_mask is not None:
                _sdpa_mask = torch.zeros_like(patch_valid_mask, dtype=x.dtype)
                _sdpa_mask.masked_fill_(~patch_valid_mask, float("-inf"))
                _sdpa_mask = _sdpa_mask.unsqueeze(1).unsqueeze(2)

            for blk in trunk_self.blocks:
                if trunk_self.grad_checkpointing and not torch.jit.is_scripting():
                    if _sdpa_mask is not None:
                        x = checkpoint(
                            blk, x, attn_mask=_sdpa_mask, use_reentrant=False
                        )
                    else:
                        x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x, attn_mask=_sdpa_mask)

            x = trunk_self.norm(x)
            return x

        self.trunk.forward_features = types.MethodType(
            _forward_features_dynamic, self.trunk
        )

    def _setup_1d_forward(self):
        """Patch forward_features for NaFlex 1D mode (SigLIP2 style)."""
        _num_prefix = getattr(self.trunk, "num_prefix_tokens", 0)

        def _forward_features_1d(
            trunk_self, x, patch_valid_mask=None, spatial_shapes=None
        ):
            from torch.utils.checkpoint import checkpoint

            conv = trunk_self.patch_embed.proj
            D = conv.weight.shape[0]
            x = torch.nn.functional.linear(
                x.to(conv.weight.dtype), conv.weight.reshape(D, -1), conv.bias
            )

            if (
                hasattr(trunk_self, "pos_embed")
                and trunk_self.pos_embed is not None
                and spatial_shapes is not None
            ):
                pos_embed = trunk_self.pos_embed
                base_n = pos_embed.shape[1]
                base_grid = int(base_n**0.5)
                pos_2d = (
                    pos_embed.reshape(1, base_grid, base_grid, -1)
                    .permute(0, 3, 1, 2)
                    .float()
                )

                B, sl, D_emb = x.shape
                pos_resized = torch.zeros(B, sl, D_emb, device=x.device, dtype=x.dtype)

                for i in range(B):
                    gh, gw = spatial_shapes[i].tolist()
                    pe = torch.nn.functional.interpolate(
                        pos_2d, size=(gh, gw), mode="bilinear", align_corners=False
                    )
                    pe = pe.squeeze(0).permute(1, 2, 0).reshape(gh * gw, -1).to(x.dtype)
                    n_patches = gh * gw
                    pos_resized[i, :n_patches] = pe
                    if n_patches < sl:
                        pos_resized[i, n_patches:] = pe[0]

                x = x + pos_resized
            elif hasattr(trunk_self, "pos_embed") and trunk_self.pos_embed is not None:
                x = x + trunk_self.pos_embed

            x = trunk_self.pos_drop(x)
            x = trunk_self.norm_pre(x)

            _sdpa_mask = None
            if patch_valid_mask is not None:
                _sdpa_mask = torch.zeros_like(patch_valid_mask, dtype=x.dtype)
                _sdpa_mask.masked_fill_(~patch_valid_mask, float("-inf"))
                _sdpa_mask = _sdpa_mask.unsqueeze(1).unsqueeze(2)

            for blk in trunk_self.blocks:
                if trunk_self.grad_checkpointing and not torch.jit.is_scripting():
                    if _sdpa_mask is not None:
                        x = checkpoint(
                            blk, x, attn_mask=_sdpa_mask, use_reentrant=False
                        )
                    else:
                        x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x, attn_mask=_sdpa_mask)

            x = trunk_self.norm(x)
            return x

        self.trunk._forward_features_1d = types.MethodType(
            _forward_features_1d, self.trunk
        )
        self._has_1d_forward = True

    def forward_patch_features(self, x):
        """Forward pass returning per-patch features (before pooling/projection)."""
        return self.trunk.forward_features(x)

    def forward(self, x, patch_valid_mask=None, spatial_shapes=None):
        if spatial_shapes is not None and getattr(self, "_has_1d_forward", False):
            patch_features = self.trunk._forward_features_1d(
                x, patch_valid_mask=patch_valid_mask, spatial_shapes=spatial_shapes
            )
        elif patch_valid_mask is not None and self._has_rope:
            patch_features = self.trunk.forward_features(x, attn_mask=patch_valid_mask)
        elif patch_valid_mask is not None:
            patch_features = self.trunk.forward_features(
                x, patch_valid_mask=patch_valid_mask
            )
        else:
            patch_features = self.trunk.forward_features(x)
        if patch_valid_mask is not None:
            mask_f = patch_valid_mask.unsqueeze(-1).to(
                patch_features.dtype
            )
            patch_features = patch_features * mask_f
        self._cached_patch_features = patch_features
        if (
            patch_valid_mask is not None
            and getattr(self.trunk, "global_pool", "") == "avg"
        ):
            pooled = patch_features.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            pooled = (
                self.trunk.fc_norm(pooled) if hasattr(self.trunk, "fc_norm") else pooled
            )
        elif (
            patch_valid_mask is not None
            and getattr(self.trunk, "attn_pool", None) is not None
        ):
            attn_mask = torch.zeros(
                patch_valid_mask.shape,
                dtype=patch_features.dtype,
                device=patch_features.device,
            )
            attn_mask.masked_fill_(~patch_valid_mask.bool(), float("-inf"))
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            pooled = self.trunk.attn_pool(patch_features, attn_mask=attn_mask)
            pooled = (
                self.trunk.fc_norm(pooled) if hasattr(self.trunk, "fc_norm") else pooled
            )
        else:
            pooled = self.trunk.forward_head(patch_features)
        pooled = self.head(pooled)
        if self.output_tokens:
            return pooled, patch_features
        return pooled
