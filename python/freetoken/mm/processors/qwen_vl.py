"""Qwen VL family image/video processor (Qwen2-VL through Qwen3.8): patch-grid ViT with a spatial merger, 3-axis rope positions.

A video arrives as one blob and becomes ONE item: evenly sampled frames fed to the checkpoint's video processor, whose ``video_grid_thw`` carries ``t > 1``. The vision tower already consumes a ``t > 1`` grid, so a video is just a temporal extension of the same patch-grid path.
"""

from __future__ import annotations

import io
import struct
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash

# Total post-merge patch budget one clip may consume. The frame cap is derived from it: a
# small frame fits many frames, a large frame fits few, so the clip's token count stays bounded
# regardless of resolution instead of a fixed count.
_PATCH_BUDGET = 32_768
_PATCH = 16
_FRAMES_CEILING = 512


def _frame_cap(h_px: int, w_px: int) -> int:
    patches = -(-h_px // _PATCH) * -(-w_px // _PATCH)  # ceil on each axis
    return max(2, min(_FRAMES_CEILING, _PATCH_BUDGET // max(1, patches)))


def _sample_indices(count: int, cap: int) -> list[int]:
    """Evenly-spaced, inclusive of both ends; returns every index when the clip already fits."""
    if count <= cap:
        return list(range(count))
    if cap <= 1:
        return [0]
    return [round(i * (count - 1) / (cap - 1)) for i in range(cap)]


class QwenVLMMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        self.image_token_id = hf_config.image_token_id
        # image and video parts can carry different pad ids; match either as the vision span,
        # but always expand to image_token_id so the single-id embed mask covers every media span.
        vid = getattr(hf_config, "video_token_id", None)
        vids = {self.image_token_id} if vid is None else {self.image_token_id, vid}
        self.placeholder = [[t] for t in sorted(vids)]
        self.merge = vc.spatial_merge_size
        self.pixels_per_token = (vc.patch_size * vc.spatial_merge_size) ** 2
        self.patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        self.temporal_patch = max(1, int(vc.temporal_patch_size))
        self.is_mrope = "mrope_section" in hf_config.text_config.rope_parameters
        self._video_processor_instance: Any = None

    def _video_processor(self) -> Any:
        """The checkpoint's video processor on first use; a still-only checkpoint falls back to the image processor."""
        with self._image_processor_lock:
            if self._video_processor_instance is None:
                from transformers import AutoVideoProcessor

                try:
                    self._video_processor_instance = AutoVideoProcessor.from_pretrained(self.model_path)
                except Exception:  # noqa: BLE001 -- no dedicated video processor: reuse the image processor per-frame
                    self._video_processor_instance = self._image_processor()
            return self._video_processor_instance

    def get_mm_processor_kwargs(self, mm: MultimodalConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if mm.image_min_tokens is not None or mm.image_max_tokens is not None:
            # the budget travels as pixel areas in size.shortest_edge / longest_edge; bare min/max_pixels kwargs are ignored
            size = dict(self._image_processor().size)
            if mm.image_min_tokens is not None:
                size["shortest_edge"] = mm.image_min_tokens * self.pixels_per_token
            if mm.image_max_tokens is not None:
                size["longest_edge"] = mm.image_max_tokens * self.pixels_per_token
            kwargs["size"] = size
        return {**kwargs, **mm.processor_kwargs}

    def _decode_part(self, raw: bytes) -> Any:
        """A still becomes one RGB image; a multi-frame clip becomes evenly-spaced RGB frames capped by the patch budget."""
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(raw))
            n_frames = getattr(img, "n_frames", 1) or 1
        except Exception:  # noqa: BLE001 -- mp4/H.264 or a codec PIL can't drive: fall back to imageio
            return self._frames_via_imageio(raw)
        if n_frames <= 1:
            return img.convert("RGB")
        w, h = img.size
        indices = _sample_indices(n_frames, _frame_cap(h, w))
        frames = []
        for i in indices:
            img.seek(i)
            frames.append(img.convert("RGB").copy())
        return frames

    def _frames_via_imageio(self, raw: bytes) -> Any:
        """Decode an mp4/H.264 blob into RGB frames via the bundled ffmpeg backend, sampled by the budget-derived cap."""
        import imageio.v3 as iio
        from PIL import Image

        stack = iio.imread(io.BytesIO(raw), extension=".mp4", index=None)  # (n, h, w, c)
        if stack.ndim == 3:
            return Image.fromarray(stack).convert("RGB")
        n, h, w = stack.shape[0], stack.shape[1], stack.shape[2]
        return [Image.fromarray(stack[i]).convert("RGB") for i in _sample_indices(n, _frame_cap(h, w))]

    def _encode_video(self, frames: list[Any], kwargs: dict[str, Any]) -> tuple[Any, list[int]]:
        """(bf16 rows, [t, h, w]) for one clip: the video processor's temporal pairing, falling back to per-frame stills."""
        try:
            import numpy as np

            arr = np.stack([np.asarray(f) for f in frames])
            out = self._video_processor()(videos=[arr], **kwargs)
            grid = out.get("video_grid_thw")
            if grid is None:
                grid = out.get("image_grid_thw")
            feature = out["pixel_values_videos" if "pixel_values_videos" in out else "pixel_values"]
            if grid is None:
                raise ValueError("video processor returned no grid")
            return feature.to(torch.bfloat16), grid[0].tolist()
        except Exception:  # noqa: BLE001 -- no working videos= entry point: stack each frame as a still
            return self._stack_frames(frames, kwargs)

    def _stack_frames(self, frames: list[Any], kwargs: dict[str, Any]) -> tuple[Any, list[int]]:
        """Fallback: encode every frame as a still and stack the patch rows frame-major into one grid."""
        processor = self._image_processor()
        rows: list[Any] = []
        hw: tuple[int, int] | None = None
        for fr in frames:
            out = processor(images=fr, **kwargs)
            _, h, w = (int(v) for v in out["image_grid_thw"][0].tolist())
            if hw is None:
                hw = (h, w)
            elif hw != (h, w):
                raise ValueError(f"clip frames resized to differing grids {hw} vs {(h, w)}")
            rows.append(out["pixel_values"].to(torch.bfloat16))
        return torch.cat(rows, dim=0), [len(rows), hw[0], hw[1]]  # type: ignore[index]

    def process(self, parts: list[Any]) -> list[MMItem]:
        processor = self._image_processor()
        kwargs = self.get_mm_processor_kwargs(self.mm)
        items: list[MMItem] = []
        for part in parts:
            frames = part if isinstance(part, list) else [part]
            if len(frames) > 1:
                feature, grid = self._encode_video(frames, kwargs)
            else:
                out = processor(images=frames[0], **kwargs)
                feature = out["pixel_values"].to(torch.bfloat16)
                grid = out["image_grid_thw"][0].tolist()
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            if t * h * w > _PATCH_BUDGET:
                raise ValueError(
                    f"media grid {t}x{h}x{w} ({t * h * w} patches) exceeds the "
                    f"{_PATCH_BUDGET}-patch budget"
                )
            modality = "video" if t > 1 else "image"
            # hash the bf16 wire buffer: half the bytes, and the exact tensor the cache and radix key agree on
            item_hash = content_hash(feature, struct.pack("<3i", t, h, w))
            items.append(
                MMItem(
                    modality=modality,
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=feature,
                    model_specific_data={"grid_thw": [t, h, w]},
                )
            )
        return items

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        t, h, w = item.grid_thw
        n_tokens = (t * h * w) // (self.merge * self.merge)
        return PromptReplacement([self.image_token_id] * n_tokens)

    def image_grid(self, item: MMItem) -> tuple[int, int]:
        _, h, w = item.grid_thw
        return h // self.merge, w // self.merge

    def positions(self, length: int, items: list[MMItem]) -> tuple[torch.Tensor, int] | None:
        if not self.is_mrope:
            return None
        return _mm_positions(length, self._blocks(items))

    def _blocks(self, items: list[MMItem]) -> list[tuple[int, int, int, int, int]]:
        blocks = []
        for item in items:
            ((start, end),) = item.offsets
            t, h, w = item.grid_thw
            blocks.append((start, end - start, t, h // self.merge, w // self.merge))
        return blocks

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        merge = self.merge
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 1]],
            feature=torch.zeros(merge * merge, self.patch_dim, dtype=dtype, device=device),
            model_specific_data={"grid_thw": [1, merge, merge]},
        )]


def _mm_positions(
    length: int, blocks: list[tuple[int, int, int, int, int]]
) -> tuple[torch.Tensor, int]:
    """t/h/w positions: text advances all three rows together; a visual block freezes its
    frame index on the t axis while spreading h/w over the merged grid, and a multi-frame
    block advances t by the frame index. The running position then jumps by the block's
    largest axis extent (HF's get_rope_index rule), which is what makes decode positions
    continue cleanly after the block."""
    pos = torch.empty((3, length), dtype=torch.int32)
    st = 0  # running position shared by all three axes between blocks
    cursor = 0
    for offset, n_tokens, t, h_m, w_m in blocks:
        run = offset - cursor
        if run:
            text = torch.arange(st, st + run, dtype=torch.int32)
            pos[:, cursor:offset] = text
            st += run
            cursor = offset
        hw = h_m * w_m  # merged rows per frame; n_tokens == t * hw for a clean grid
        idx = torch.arange(n_tokens, dtype=torch.int32)
        frame = idx // hw
        row_in = (idx % hw) // w_m
        col_in = idx % w_m
        pos[0, cursor : cursor + n_tokens] = st + frame
        pos[1, cursor : cursor + n_tokens] = st + row_in
        pos[2, cursor : cursor + n_tokens] = st + col_in
        st += max(h_m, w_m)
        cursor += n_tokens
    run = length - cursor
    if run:
        pos[:, cursor:] = torch.arange(st, st + run, dtype=torch.int32)
        st += run
    delta = int(pos.max().item()) + 1 - length
    return pos, delta


__all__ = ["QwenVLMMProcessor"]
