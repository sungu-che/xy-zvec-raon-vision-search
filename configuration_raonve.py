"""Raon-VisionEncoder configuration."""

from transformers import PretrainedConfig


class RaonVEVisionConfig(PretrainedConfig):
    model_type = "raon_ve_vision"

    def __init__(
        self,
        image_size=256,
        timm_model_name="vit_so400m_patch16_siglip_256",
        timm_model_pretrained=False,
        timm_pool="map",
        timm_proj="none",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.timm_model_name = timm_model_name
        self.timm_model_pretrained = timm_model_pretrained
        self.timm_pool = timm_pool
        self.timm_proj = timm_proj


class RaonVETextConfig(PretrainedConfig):
    model_type = "raon_ve_text"

    def __init__(
        self,
        context_length=64,
        vocab_size=256000,
        width=1152,
        heads=16,
        layers=27,
        mlp_ratio=3.7362,
        no_causal_mask=True,
        proj_bias=True,
        pool_type="last",
        hf_tokenizer_name="timm/ViT-SO400M-16-SigLIP2-256",
        tokenizer_kwargs=None,
        norm_kwargs=None,
        act_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.layers = layers
        self.mlp_ratio = mlp_ratio
        self.no_causal_mask = no_causal_mask
        self.proj_bias = proj_bias
        self.pool_type = pool_type
        self.hf_tokenizer_name = hf_tokenizer_name
        self.tokenizer_kwargs = tokenizer_kwargs or {"clean": "canonicalize"}
        self.norm_kwargs = norm_kwargs or {"eps": 1e-6}
        self.act_kwargs = act_kwargs or {"approximate": "tanh"}


class RaonVEConfig(PretrainedConfig):
    model_type = "raon_ve"
    is_composition = True

    def __init__(
        self,
        embed_dim=1152,
        init_logit_bias=-10,
        vision_config=None,
        text_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.init_logit_bias = init_logit_bias

        if isinstance(vision_config, dict):
            self.vision_config = RaonVEVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = RaonVEVisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = RaonVETextConfig(**text_config)
        elif text_config is None:
            self.text_config = RaonVETextConfig()
        else:
            self.text_config = text_config

    def to_dict(self):
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output
