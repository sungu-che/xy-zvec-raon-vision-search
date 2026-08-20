# Originally from OpenCLIP (https://github.com/mlfoundations/open_clip)

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

from .timm_model import TimmModel
from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    TextTransformer,
    text_global_pool,
)
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    no_ln_pre: bool = False
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    block_type: Optional[str] = None
    qk_norm: bool = False
    scaled_cosine_attn: bool = False
    scale_heads: bool = False
    scale_attn_inner: bool = False
    scale_attn: bool = False
    scale_fc: bool = False

    timm_model_name: Optional[str] = None
    timm_model_pretrained: bool = False
    timm_pool: str = "avg"
    timm_proj: str = "linear"
    timm_proj_bias: bool = False
    timm_drop: float = 0.0
    timm_drop_path: Optional[float] = None
    timm_use_rope: bool = False
    timm_rope_keep_ape: bool = False
    timm_dynamic_img_size: bool = False
    timm_norm_pre: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_mode: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None
    embed_cls: bool = False
    pad_id: int = 0
    eos_id: int = 2
    no_causal_mask: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = "argmax"
    proj_bias: bool = False
    proj_type: str = "linear"
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    block_type: Optional[str] = None
    qk_norm: bool = False
    scaled_cosine_attn: bool = False
    scale_heads: bool = False
    scale_attn_inner: bool = False
    scale_attn: bool = False
    scale_fc: bool = False

    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = "mlp"
    hf_pooler_type: str = "mean_pooler"


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    if not vision_cfg.timm_model_name:
        raise ValueError(
            "Only TimmModel-based vision towers are supported in raon-vision-encoder. "
            "Please set timm_model_name in vision_cfg."
        )

    visual = TimmModel(
        vision_cfg.timm_model_name,
        pretrained=vision_cfg.timm_model_pretrained,
        pool=vision_cfg.timm_pool,
        proj=vision_cfg.timm_proj,
        proj_bias=vision_cfg.timm_proj_bias,
        drop=vision_cfg.timm_drop,
        drop_path=vision_cfg.timm_drop_path,
        patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
        init_values=vision_cfg.ls_init_value,
        qk_norm=vision_cfg.qk_norm,
        use_rope=vision_cfg.timm_use_rope,
        rope_keep_ape=vision_cfg.timm_rope_keep_ape,
        dynamic_img_size=vision_cfg.timm_dynamic_img_size,
        norm_pre=vision_cfg.timm_norm_pre,
        embed_dim=embed_dim,
        image_size=vision_cfg.image_size,
        output_tokens=vision_cfg.output_tokens,
    )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )
    if text_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
    if text_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **text_cfg.act_kwargs)

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        mlp_ratio=text_cfg.mlp_ratio,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        no_causal_mask=text_cfg.no_causal_mask,
        pad_id=text_cfg.pad_id,
        eos_id=text_cfg.eos_id,
        pool_type=text_cfg.pool_type,
        proj_type=text_cfg.proj_type,
        proj_bias=text_cfg.proj_bias,
        output_tokens=text_cfg.output_tokens,
        act_layer=act_layer,
        norm_layer=norm_layer,
        block_type=text_cfg.block_type,
        qk_norm=text_cfg.qk_norm,
        scaled_cosine_attn=text_cfg.scaled_cosine_attn,
        scale_heads=text_cfg.scale_heads,
        scale_attn_inner=text_cfg.scale_attn_inner,
        scale_attn=text_cfg.scale_attn,
        scale_fc=text_cfg.scale_fc,
    )
    return text


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def encode_image(
        self, pixel_values, normalize: bool = False, pixel_attention_mask=None, spatial_shapes=None
    ):
        kwargs = {}
        if pixel_attention_mask is not None:
            kwargs["patch_valid_mask"] = pixel_attention_mask
        if spatial_shapes is not None:
            kwargs["spatial_shapes"] = spatial_shapes
        features = self.visual(pixel_values, **kwargs) if kwargs else self.visual(pixel_values)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, input_ids, normalize: bool = False):
        features = self.text(input_ids)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(pixel_values=image, normalize=True)
        text_features = self.encode_text(input_ids=text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self, image=None, text=None, patch_valid_mask=None, spatial_shapes=None
    ):
        image_features = (
            self.encode_image(
                pixel_values=image,
                normalize=True,
                pixel_attention_mask=patch_valid_mask,
                spatial_shapes=spatial_shapes,
            )
            if image is not None
            else None
        )
        text_features = (
            self.encode_text(input_ids=text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return (
                image_features,
                text_features,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        return image_features, text_features, self.logit_scale.exp()
