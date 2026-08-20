# Originally from OpenCLIP (https://github.com/mlfoundations/open_clip)

from collections import OrderedDict
import math
from typing import Callable, Optional, Type, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scaled_cosine: bool = False,
        scale_heads: bool = False,
        inner_norm: bool = False,
        logit_scale_max: float = math.log(1.0 / 0.01),
        norm_layer: Type[nn.Module] = LayerNormFp32,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert not (scaled_cosine and qk_norm), (
            "Cannot activate both scaled cosine and QK normalization"
        )
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max
        self.use_fsdpa = hasattr(nn.functional, "scaled_dot_product_attention")

        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if qk_norm:
            self.ln_q = norm_layer(self.head_dim)
            self.ln_k = norm_layer(self.head_dim)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
        else:
            self.logit_scale = None

        self.attn_drop = nn.Dropout(attn_drop)

        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None

        if inner_norm:
            self.ln_inner = norm_layer(dim)
        else:
            self.ln_inner = nn.Identity()

        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        N, L, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(N, L, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(N, L, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(N, L, self.num_heads, -1).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.reshape(N, self.num_heads, L, L)
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            else:
                attn_mask = attn_mask.to(dtype=q.dtype)

        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2)
            )
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn * logit_scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            q = self.ln_q(q)
            k = self.ln_k(k)
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x * self.head_scale
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.ln_inner(x)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, "int8_original_dtype"):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x,
            k_x,
            v_x,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(
                q_x=self.ln_1(q_x),
                k_x=k_x,
                v_x=v_x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        qk_norm: bool = False,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn_inner: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        assert batch_first, "batch_first must be True for CustomResidualAttentionBlock"

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            qk_norm=qk_norm,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            inner_norm=scale_attn_inner,
            norm_layer=norm_layer,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("ln", norm_layer(mlp_width) if scale_fc else nn.Identity()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, "int8_original_dtype"):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        batch_first: bool = True,
        block_type: Optional[str] = None,
        qk_norm: bool = False,
        scaled_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn_inner: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        if block_type is None:
            if any(
                [
                    qk_norm,
                    scaled_cosine_attn,
                    scale_heads,
                    scale_attn_inner,
                    scale_attn,
                    scale_fc,
                ]
            ):
                block_type = "custom"
            else:
                block_type = "default"

        if block_type == "custom":
            self.resblocks = nn.ModuleList(
                [
                    CustomResidualAttentionBlock(
                        width,
                        heads,
                        mlp_ratio,
                        ls_init_value=ls_init_value,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        qk_norm=qk_norm,
                        scale_cosine_attn=scaled_cosine_attn,
                        scale_heads=scale_heads,
                        scale_attn_inner=scale_attn_inner,
                        scale_attn=scale_attn,
                        scale_fc=scale_fc,
                        batch_first=batch_first,
                    )
                    for _ in range(layers)
                ]
            )
        else:
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock(
                        width,
                        heads,
                        mlp_ratio,
                        ls_init_value=ls_init_value,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        batch_first=batch_first,
                    )
                    for _ in range(layers)
                ]
            )

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].get_weight_dtype()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def text_global_pool(
    x: torch.Tensor,
    text: Optional[torch.Tensor] = None,
    pool_type: str = "argmax",
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if pool_type == "first":
        pooled = x[:, 0]
    elif pool_type == "last":
        pooled = x[:, -1]
    elif pool_type == "argmax":
        assert text is not None
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    elif pool_type == "eos":
        assert text is not None
        assert eos_token_id is not None
        idx = (text == eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), idx]
    else:
        pooled = x

    return pooled


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: Optional[int] = 512,
        embed_cls: bool = False,
        no_causal_mask: bool = False,
        use_pad_mask: bool = False,
        correct_cls_mask: bool = False,
        pad_id: int = 0,
        eos_id: int = 2,
        pool_type: str = "argmax",
        proj_type: str = "linear",
        proj_bias: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        output_tokens: bool = False,
        block_type: Optional[str] = None,
        qk_norm: bool = False,
        scaled_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn_inner: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "eos", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.pool_type = pool_type
        self.use_pad_mask = use_pad_mask and no_causal_mask
        self.correct_cls_mask = correct_cls_mask

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=block_type,
            qk_norm=qk_norm,
            scaled_cosine_attn=scaled_cosine_attn,
            scale_heads=scale_heads,
            scale_attn_inner=scale_attn_inner,
            scale_attn=scale_attn,
            scale_fc=scale_fc,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )

        if proj_type == "none" or not output_dim:
            self.text_projection = None
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(
                    self.text_projection.weight, std=self.transformer.width**-0.5
                )
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_causal_mask(self):
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def _build_additive_mask(self, text, seq_len, dtype):
        valid = text != self.pad_id
        if self.cls_emb is not None:
            cls_valid = valid.new_ones(valid.size(0), 1)
            valid = torch.cat(
                [valid, cls_valid] if self.correct_cls_mask else [cls_valid, valid], 1
            )
        key_mask = valid.unsqueeze(1).expand(-1, seq_len, -1)
        additive = torch.zeros_like(key_mask, dtype=dtype)
        additive.masked_fill_(~key_mask, float("-inf"))
        additive = additive.repeat_interleave(self.heads, 0)
        return additive

    def _embeds(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        B, seq_len = text.shape
        x = self.token_embedding(text).to(cast_dtype)
        if self.cls_emb is not None:
            x = torch.cat([x, _expand_token(self.cls_emb, x.size(0))], 1)
            seq_len += 1
        attn_mask = self.attn_mask
        if self.use_pad_mask or self.cls_emb is not None:
            add_mask = self._build_additive_mask(text, seq_len, x.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[:seq_len, :seq_len].unsqueeze(0) + add_mask
            else:
                attn_mask = add_mask
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        return x, attn_mask

    def forward(self, text):
        x, attn_mask = self._embeds(text)
        x = self.transformer(x, attn_mask=attn_mask)
        if self.cls_emb is not None:
            pooled = text_global_pool(x, pool_type="last")
            pooled = self.ln_final(pooled)
            tokens = x[:, :-1]
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(
                x,
                text,
                pool_type=self.pool_type,
                eos_token_id=getattr(self, "eos_id", None),
            )
            tokens = x
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled
