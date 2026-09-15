from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    mm_mrope: torch.Tensor | None = None  # [prompt_len, 3] int32 cpu (see Req.mm_mrope)
    # P4: content-addressed prefix-cache key (input_ids with image_pad runs -> payload-hash
    # id). match_req / cache_req key mm match+insert on this so identical images reuse KV.
    mm_cache_key: torch.Tensor | None = None
    # mm RAM tier: host GDN state snapshot from CacheManager.restore_mm (set at admission),
    # written into the request's live slot on the first prefill forward.
    mm_gdn: tuple | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
