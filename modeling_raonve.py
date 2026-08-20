"""Raon-VisionEncoder model."""

import importlib
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from configuration_raonve import RaonVEConfig


_raon_repo_id = None

def set_repo_id(repo_id):
    global _raon_repo_id
    _raon_repo_id = repo_id

def _ensure_raon_package():
    """Import raon_vision_encoder, downloading from HF Hub if needed."""
    try:
        clip_mod = importlib.import_module("raon_vision_encoder.clip")
        return clip_mod.CustomTextCLIP
    except (ImportError, ModuleNotFoundError):
        pass

    from huggingface_hub import snapshot_download
    repo_id = _raon_repo_id or "KRAFTON/Raon-VisionEncoder"
    repo_dir = snapshot_download(repo_id, allow_patterns=["raon_vision_encoder/**"])
    sys.path.insert(0, repo_dir)

    for key in list(sys.modules.keys()):
        if key.startswith("raon_vision_encoder"):
            del sys.modules[key]

    clip_mod = importlib.import_module("raon_vision_encoder.clip")
    return clip_mod.CustomTextCLIP


class RaonVEPreTrainedModel(PreTrainedModel):
    config_class = RaonVEConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        pass


class RaonVEModel(RaonVEPreTrainedModel):
    config_class = RaonVEConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        set_repo_id(str(pretrained_model_name_or_path))
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def __init__(self, config: RaonVEConfig):
        super().__init__(config)

        vision_cfg = {
            "image_size": config.vision_config.image_size,
            "timm_model_name": config.vision_config.timm_model_name,
            "timm_model_pretrained": config.vision_config.timm_model_pretrained,
            "timm_pool": config.vision_config.timm_pool,
            "timm_proj": config.vision_config.timm_proj,
        }
        text_cfg = {
            "context_length": config.text_config.context_length,
            "vocab_size": config.text_config.vocab_size,
            "width": config.text_config.width,
            "heads": config.text_config.heads,
            "layers": config.text_config.layers,
            "mlp_ratio": config.text_config.mlp_ratio,
            "no_causal_mask": config.text_config.no_causal_mask,
            "proj_bias": config.text_config.proj_bias,
            "pool_type": config.text_config.pool_type,
            "hf_tokenizer_name": config.text_config.hf_tokenizer_name,
            "tokenizer_kwargs": config.text_config.tokenizer_kwargs,
            "norm_kwargs": config.text_config.norm_kwargs,
            "act_kwargs": config.text_config.act_kwargs,
        }

        CustomTextCLIP = _ensure_raon_package()
        inner = CustomTextCLIP(
            embed_dim=config.embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            init_logit_bias=config.init_logit_bias,
        )

        self.visual = inner.visual
        self.text = inner.text
        self.logit_scale = inner.logit_scale
        self.logit_bias = inner.logit_bias

        # Enable NaFlex by default
        self.visual._setup_1d_forward()

        self.post_init()

    def encode_image(self, pixel_values, pixel_attention_mask=None, spatial_shapes=None):
        """Encode images to normalized feature vectors [B, 1152].
        Pass the output of processor(images=...) directly via **inputs.
        """
        kwargs = {}
        if pixel_attention_mask is not None:
            kwargs["patch_valid_mask"] = pixel_attention_mask
        if spatial_shapes is not None:
            kwargs["spatial_shapes"] = spatial_shapes
        features = self.visual(pixel_values, **kwargs) if kwargs else self.visual(pixel_values)
        return F.normalize(features, dim=-1)

    def encode_text(self, input_ids):
        """Encode text to normalized feature vectors [B, 1152].
        Pass the output of processor(text=...) directly via **inputs.
        """
        features = self.text(input_ids)
        return F.normalize(features, dim=-1)

    def forward(self, pixel_values=None, input_ids=None, pixel_attention_mask=None, spatial_shapes=None):
        image_features = None
        text_features = None

        if pixel_values is not None:
            image_features = self.encode_image(
                pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
        if input_ids is not None:
            text_features = self.encode_text(input_ids)

        output = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale,
            "logit_bias": self.logit_bias,
        }
        return output

    @staticmethod
    def get_processor(pretrained_model_name_or_path, **kwargs):
        """Get the processor for this model."""
        return RaonVEProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)


class RaonVEProcessor:
    """Image and text processor for Raon-VisionEncoder.

    Preprocesses images into NaFlex patch sequences and tokenizes text.

    Args:
        max_num_patches: Maximum number of patches per image (controls resolution).
            Higher values preserve more detail. Default: 256.
    """

    DEFAULT_MAX_PATCHES = 256

    def __init__(self, patch_size=16, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), tokenizer=None):
        from torchvision import transforms as T
        self.patch_size = patch_size
        self.mean, self.std = mean, std
        self.tokenizer = tokenizer
        self._post = T.Compose([T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path as _Path
        if _Path(pretrained_model_name_or_path).is_dir():
            cfg_path = _Path(pretrained_model_name_or_path) / "config.json"
        else:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        v = cfg.get("vision_config", {}); t = cfg.get("text_config", {})
        ps = 16
        for part in v.get("timm_model_name", "").split("_"):
            if part.startswith("patch") and part[5:].isdigit():
                ps = int(part[5:]); break
        tokenizer = None
        if t.get("hf_tokenizer_name"):
            _ensure_raon_package()
            tok_mod = importlib.import_module("raon_vision_encoder.tokenizer")
            tokenizer = tok_mod.HFTokenizer(
                t["hf_tokenizer_name"], context_length=t.get("context_length", 64),
                tokenizer_mode=t.get("tokenizer_mode"), token=False,
                **t.get("tokenizer_kwargs", {}),
            )
        return cls(patch_size=ps, tokenizer=tokenizer)

    def __call__(self, images=None, text=None, max_num_patches=None, return_tensors="pt"):
        """Process images and/or text.

        Args:
            images: PIL Image or list of PIL Images.
            text: String or list of strings.
            max_num_patches: Resolution budget (default: 256). Higher = more detail.

        Returns:
            Dict with 'pixel_values', 'pixel_attention_mask', 'spatial_shapes' for images
            and/or 'input_ids' for text.
        """
        from PIL import Image
        result = {}
        if images is not None:
            mnp = max_num_patches or self.DEFAULT_MAX_PATCHES
            _ensure_raon_package()
            transform_mod = importlib.import_module("raon_vision_encoder.transform")
            get_size = transform_mod.get_image_size_for_max_num_patches
            imgs = [images] if isinstance(images, Image.Image) else images
            ps = self.patch_size
            all_p, all_m, all_s = [], [], []
            for img in imgs:
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGBA")
                    _bg = Image.new("RGB", img.size, (255, 255, 255))
                    _bg.paste(img, mask=img.split()[-1])
                    img = _bg
                else:
                    img = img.convert("RGB")
                w, h = img.size
                th, tw = get_size(h, w, ps, mnp)
                t = self._post(img.resize((tw, th), Image.BICUBIC))
                gh, gw = th // ps, tw // ps
                n = gh * gw
                # [C, gh, ps, gw, ps] -> [gh, gw, C, ps, ps] -> [n, C*ps*ps]
                patches = t.reshape(3, gh, ps, gw, ps).permute(1,3,0,2,4).reshape(n, 3*ps*ps)
                padded = torch.zeros(mnp, ps*ps*3); padded[:n] = patches
                mask = torch.zeros(mnp, dtype=torch.bool); mask[:n] = True
                all_p.append(padded); all_m.append(mask)
                all_s.append(torch.tensor([gh, gw]))
            result["pixel_values"] = torch.stack(all_p)
            result["pixel_attention_mask"] = torch.stack(all_m)
            result["spatial_shapes"] = torch.stack(all_s)
        if text is not None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized.")
            result["input_ids"] = self.tokenizer([text] if isinstance(text, str) else text)
        return result
