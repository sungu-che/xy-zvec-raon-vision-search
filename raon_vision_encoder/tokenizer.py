# Originally from OpenCLIP (https://github.com/mlfoundations/open_clip)

import html
import os
import string
from typing import List, Optional, Union
import warnings

try:
    import ftfy
except ImportError:
    ftfy = None
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONTEXT_LENGTH = 77


def basic_clean(text):
    if ftfy is not None:
        text = ftfy.fix_text(text)
    else:
        text
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = " ".join(text.split())
    text = text.strip()
    return text


def _clean_canonicalize(x):
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == "canonicalize":
        return _clean_canonicalize
    elif type == "lower":
        return _clean_lower
    elif type == "whitespace":
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def canonicalize_text(
    text,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed)."""
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


class HFTokenizer:
    """HuggingFace tokenizer wrapper with support for custom tokenization modes"""

    def __init__(
        self,
        tokenizer_name: str,
        context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
        clean: str = "whitespace",
        strip_sep_token: bool = False,
        language: Optional[str] = None,
        cache_dir: Optional[str] = None,
        tokenizer_mode: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer_mode = tokenizer_mode or ""
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir, **kwargs
        )

        set_lang_fn = getattr(self.tokenizer, "set_src_lang_special_tokens", None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(
        self, texts: Union[str, List[str]], context_length: Optional[int] = None
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, (
            "Please set a valid context length in class init or call."
        )

        texts = [self.clean_fn(text) for text in texts]

        if self.tokenizer_mode == "clips":
            return self._clips_tokenize(texts, context_length)
        else:
            output = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=context_length,
                padding="max_length",
                truncation=True,
            )
            input_ids = output.input_ids

            if self.strip_sep_token:
                input_ids = torch.where(
                    input_ids == self.tokenizer.sep_token_id,
                    torch.zeros_like(input_ids),
                    input_ids,
                )

            return input_ids

    def set_language(self, src_lang):
        if hasattr(self, "set_lang_fn"):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn("Cannot set language for the tokenizer.")

    def _clips_tokenize(self, texts: List[str], context_length: int) -> torch.Tensor:
        encoded_outputs = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )

        encoded = []
        for tokens in encoded_outputs["input_ids"]:
            tokens = tokens[: context_length - 3]
            tokens = (
                [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            )
            encoded.append(tokens)

        result = torch.zeros(len(encoded), context_length, dtype=torch.long)
        for i, tokens in enumerate(encoded):
            padded_tokens = self._pad_and_add_class_token(
                tokens,
                max_length=context_length,
                pad_token_id=self.tokenizer.pad_token_id,
                cls_token_id=self.tokenizer.cls_token_id,
            )
            result[i, : len(padded_tokens)] = torch.tensor(padded_tokens)

        return result

    def _pad_and_add_class_token(
        self,
        tokens: List[int],
        max_length: int,
        pad_token_id: int = 0,
        cls_token_id: int = 101,
    ) -> List[int]:
        if len(tokens) > max_length - 1:
            tokens = tokens[: max_length - 1]
        if len(tokens) < max_length - 1:
            tokens = tokens + [pad_token_id] * (max_length - 1 - len(tokens))
        tokens = tokens + [cls_token_id]
        return tokens
