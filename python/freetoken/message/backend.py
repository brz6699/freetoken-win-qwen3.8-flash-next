from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings (GPU tensor). Only used by
    # the in-process offline path; remains None for the (serialized) online path, where
    # the SCHEDULER produces it by encoding msg.mm_data_path with the vision tower.
    mm_embeds: torch.Tensor | None = None
    # M-RoPE table for an image-bearing prompt: flat [prompt_len*3] int32 (t/h/w per
    # token). The online serializer carries 1-D tensors only, so the tokenizer worker
    # flattens and the scheduler reshapes with view(-1, 3). None = pure text: every rope
    # consumer stays on the unchanged 1-D logical-positions path.
    mm_mrope: torch.Tensor | None = None
    # Staged pixel payload path ({"pixel_values", "image_grid_thw"} .pt file) the
    # tokenizer worker wrote as the cross-process side channel for the tens of MB that
    # do not belong in a serialized message. The scheduler loads it, encodes, deletes it.
    mm_data_path: str | None = None
    # P4: content-addressed PREFIX-CACHE key for an image-bearing prompt -- a copy of
    # input_ids with every image_pad position replaced by a payload-hash-derived id.
    # The radix tree keys on token ids, and image_pad ids are identical across images
    # while their KV differs, so matching on raw ids would serve the wrong image's KV.
    # Keying the span by the content id makes identical images reuse KV and different
    # images never false-match. Same length as input_ids, int32, None for text-only
    # (every consumer stays on the unchanged raw-id path) and for offline mm (which
    # precomputes mm_embeds in-process and carries no payload path to hash).
    mm_cache_key: torch.Tensor | None = None


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
