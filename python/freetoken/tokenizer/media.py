"""CPU-side media pipeline for image-bearing chat requests (tokenizer worker).

The worker owns everything that is pure-CPU about a vision request: decoding the
image (data: URI / http(s) / local path), running the checkpoint's HF image
processor, expanding the chat template's single image-pad placeholder to the
grid's token count, computing the interleaved M-RoPE position table (the exact
algorithm of HF ``Qwen4ExpTextModel.get_rope_index``), and staging the pixel
payload as a .pt side channel (the serializer only carries 1-D CPU tensors; the
pixels are tens of MB and belong in a file named by content hash, so the P4
radix-key work can hash the same bytes). The SCHEDULER process owns the vision
tower and encodes the staged payload when the UserMsg arrives.

Token layout invariant (mirrors HF): the prompt contains, per image, exactly
``t * (h/merge) * (w/merge)`` consecutive image-pad tokens; the mrope table
covers the WHOLE prompt ([P, 3]) with the three channels equal to the logical
index on text tokens, and decode steps past the prompt advance all three
channels by one per token from ``table.max() + 1`` (HF's rope_deltas rule).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from typing import Any, Sequence

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# An image whose DECODED bytes exceed this is rejected (client error), never a
# worker OOM. 32 MiB of raw pixels is far beyond the processor's longest_edge
# cap anyway; it bounds only the decode buffer.
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_FETCH_TIMEOUT_S = 20

# Frame cap for one video: how many sampled frames fit the engine's vision
# patch budget (the same 32768-patch ceiling prepare() enforces). n frames
# cost n * patches_per_frame in the worst encode path -- the per-frame stacking
# fallback gives each frame its own row of t, and only the videos= pairing path
# halves t -- so the cap solves the inequality against the UNHALVED cost:
# small frames sample densely, large frames fewer. Ceiling keeps GIF/animation
# decode bounded.
_PATCH_BUDGET = 32_768
_PATCH = 16
_FRAMES_CEILING = 512


def _frame_cap(h_px: int, w_px: int) -> int:
    """Frame-count cap from the frame's pixel size and the patch budget."""
    patches = -(-h_px // _PATCH) * (-(-w_px // _PATCH))
    return max(2, min(_FRAMES_CEILING, _PATCH_BUDGET // max(1, patches)))


def _read_raw(url: str) -> bytes:
    """Bytes behind a media reference: data URI, http(s) fetch, or local file."""
    if url.startswith("data:"):
        _, _, b64 = url.partition(",")
        raw = base64.b64decode(b64, validate=False)
    elif url.startswith("http://") or url.startswith("https://"):
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as resp:
            raw = resp.read()
    else:
        if not os.path.isfile(url):
            raise ValueError(
                f"image is neither a data:/http(s) URL nor an existing file: {url[:200]}"
            )
        with open(url, "rb") as fh:
            raw = fh.read()
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MiB after decode: {len(raw)} bytes")
    return raw


def fetch_image(url: str) -> Any:
    """Decode an image reference to a PIL image (RGB).

    Accepts ``data:<mime>;base64,<payload>`` URIs, ``http(s)://`` URLs, and
    plain filesystem paths (local files). Raises ValueError (client error,
    surfaced per-request) on anything undecodable.
    """
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(_read_raw(url)))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError(f"could not decode image: {exc!r}") from exc


def _sample_indices(count: int, cap: int) -> list[int]:
    """Uniform index set drawing `count` items down to `cap` (inclusive ends)."""
    if count <= cap:
        return list(range(count))
    return [round(i * (count - 1) / (cap - 1)) for i in range(cap)]


def fetch_video_frames(url: str) -> list[Any]:
    """RGB frames of one video reference: an animated image (GIF/WebP/APNG)
    exposes its frames through n_frames/seek; a still reports one frame and
    behaves exactly like fetch_image. Frames beyond the size-derived cap are
    sampled uniformly (first and last always kept)."""
    from PIL import Image, ImageOps

    raw = _read_raw(url)
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        # Pillow has no built-in mp4/H.264 path; containers it cannot
        # identify go through the imageio/ffmpeg backend below.
        return _frames_via_imageio(raw, url)
    count = int(getattr(img, "n_frames", 1) or 1)
    w_px, h_px = img.size
    frames: list[Any] = []
    for idx in _sample_indices(count, _frame_cap(int(h_px), int(w_px))):
        img.seek(idx)
        frames.append(ImageOps.exif_transpose(img).convert("RGB"))
    return frames


def _frames_via_imageio(raw: bytes, ref: str) -> list[Any]:
    """Decode an mp4-style container through imageio (bundled ffmpeg exe).
    The legacy ffmpeg backend reads from a filesystem path, so a non-path
    reference (data:/http) is spilled to a temp file first. The whole stream
    is read once, then sampled down to the frame cap."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ValueError(f"could not decode video: {exc!r}") from exc

    from PIL import Image

    src: str
    if os.path.isfile(ref):
        src = ref
    else:
        suffix = ".mp4"
        if ref.startswith("data:"):
            head = ref.split(",", 1)[0]
            mime = head.split(":", 1)[1].split(";")[0]
            if "/" in mime:
                suffix = "." + mime.split("/", 1)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(raw)
            src = fh.name
    try:
        stack = iio.imread(src, index=None)
    except Exception as exc:
        raise ValueError(f"could not decode video: {exc!r}") from exc
    finally:
        if src != ref and os.path.isfile(src):
            os.remove(src)
    if stack.ndim == 3:  # single frame (H, W, C)
        stack = stack[None]

    return [
        Image.fromarray(stack[i])
        for i in _sample_indices(int(stack.shape[0]), _frame_cap(int(stack.shape[1]), int(stack.shape[2])))
    ]


def extract_media_parts(messages: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """Ordered ``(kind, value)`` media parts across all messages (template-shaped
    parts: ``{"type": "image", "image": <url>}`` and
    ``{"type": "video", "video": <url-or-frame-list>}``). Empty = text-only
    request. One part = one placeholder in the rendered prompt, whatever number
    of frames the video value carries."""
    parts: list[tuple[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            # Mirror the chat template's branch order: 'image'/'image_url' keys
            # or type=="image" first, then the video spellings. image_url carries
            # its reference inside {"url": ...}; unwrap it here so both spellings
            # hand the rest of the pipeline a plain reference.
            if "image" in part or "image_url" in part or part.get("type") == "image":
                value = part.get("image", part.get("image_url"))
                if isinstance(value, dict):
                    value = value.get("url")
                parts.append(("image", value))
            elif "video" in part or part.get("type") == "video":
                value = part.get("video", part.get("video_url"))
                parts.append(("video", value))
    return parts


def extract_image_urls(messages: list[dict[str, Any]]) -> list[str]:
    """Flat url list of every media reference in document order (image urls and
    video frame refs). Truthiness = the multimodal gate; the ordered part
    structure is ``extract_media_parts``."""
    urls: list[str] = []
    for _kind, value in extract_media_parts(messages):
        if isinstance(value, (list, tuple)):
            urls.extend(str(v) for v in value)
        else:
            urls.append(str(value))
    return urls


def _spatial_merge(vision_config: Any) -> int:
    """The vision spatial merge size from either a config dict or a PretrainedConfig."""
    if vision_config is None:
        return 2
    if isinstance(vision_config, dict):
        value = vision_config.get("spatial_merge_size")
    else:
        value = getattr(vision_config, "spatial_merge_size", None)
    return int(value) if value else 2


class MediaPipeline:
    """Lazy HF image processor + prompt expansion + mrope table for one checkpoint."""

    def __init__(
        self, model_path: str, image_token_id: int, vision_config: Any,
        video_token_id: int | None = None,
    ) -> None:
        self._model_path = model_path
        self._image_token_id = int(image_token_id)
        # Checkpoints spell the video placeholder with its own pad id
        # (video_token_id); when one is absent the two spellings share an id.
        self._video_token_id = (
            int(video_token_id) if video_token_id is not None else int(image_token_id)
        )
        self._merge = _spatial_merge(vision_config)
        self._processor: Any | None = None
        self._processor_failed: Exception | None = None

    def _ensure_loaded(self) -> None:
        if self._processor is None and self._processor_failed is None:
            try:
                from transformers import AutoImageProcessor

                self._processor = AutoImageProcessor.from_pretrained(self._model_path)
            except Exception as exc:  # noqa: BLE001 — report per request, never kill the worker
                self._processor_failed = exc
        if self._processor is None:
            raise ValueError(f"image processor unavailable for this checkpoint: {self._processor_failed!r}")

    def _decode_frames(self, kind: str, value: Any) -> list[Any]:
        """RGB frames for one media part. An image is a single frame; a video is
        either an ordered list of frame references or one animated file, already
        capped by the fetch-side sampler."""
        if kind == "image":
            return [fetch_image(str(value))]
        refs = list(value) if isinstance(value, (list, tuple)) else [value]
        frames: list[Any] = []
        for ref in refs:
            frames.extend(fetch_video_frames(str(ref)))
        if not frames:
            raise ValueError("video part decoded no frames")
        return frames

    def _encode_frames(self, frames: list[Any]) -> "tuple[torch.Tensor, list[int]]":
        """(pixel rows, [t, h, w]) for one media part. A single frame keeps the
        exact images= path; multiple frames go through the processor's videos=
        pairing (two frames per temporal unit, HF parity), falling back to
        per-frame stills stacked frame-major when the processor exposes no
        working videos= entry point."""
        if len(frames) == 1:
            out = self._processor(images=frames[0], return_tensors="pt")
            pixels, grid = out["pixel_values"], out["image_grid_thw"]
        else:
            out = None
            grid = None
            try:
                import numpy as np

                arr = np.stack([np.asarray(f) for f in frames])
                out = self._processor(videos=[arr], return_tensors="pt")
                grid = out.get("video_grid_thw")
                if grid is None:
                    grid = out.get("image_grid_thw")
                pixels = out["pixel_values"]
            except Exception:
                out = None
            if out is None or grid is None:
                return self._stack_frame_grids(frames)
        # A single image already comes back as [t*h*w, patch_dim] (2-D); only drop a
        # leading batch dim if one is actually present.
        if pixels.dim() == 3 and pixels.shape[0] == 1:
            pixels = pixels[0]
        if grid.dim() == 2:
            grid = grid[0]
        return pixels, [int(v) for v in grid.tolist()]

    def _stack_frame_grids(self, frames: list[Any]) -> "tuple[torch.Tensor, list[int]]":
        """Fallback video encoding: process every frame as a still (images=) and
        stack the patch rows frame-major into one [n, h, w] grid. All frames must
        resize to the same (h, w); the processor decides that from frame size."""
        rows: list[torch.Tensor] = []
        hw: "tuple[int, int] | None" = None
        for f in frames:
            out = self._processor(images=f, return_tensors="pt")
            pixels = out["pixel_values"]
            grid = out["image_grid_thw"]
            if pixels.dim() == 3 and pixels.shape[0] == 1:
                pixels = pixels[0]
            if grid.dim() == 2:
                grid = grid[0]
            _t, h, w = (int(v) for v in grid.tolist())
            if hw is None:
                hw = (h, w)
            elif hw != (h, w):
                raise ValueError(f"video frames must share one resized grid; got {hw} then {(h, w)}")
            rows.append(pixels.reshape(-1, pixels.shape[-1]))
        assert hw is not None and rows
        return torch.cat(rows, 0), [len(rows), hw[0], hw[1]]

    def prepare(
        self,
        messages: list[dict[str, Any]],
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str] | None:
        """Turn an image-bearing tokenization into (ids, mrope_table, payload_path, cache_key).

        Returns None when the message carries no image parts (the caller keeps
        the plain path). ``input_ids`` is the rendered prompt's tokens (1-D
        int32, holding ONE image_pad per image); the i-th occurrence is replaced
        by the i-th image's token block. ``cache_key`` (P4) is ids with the image_pad
        runs content-addressed (payload-hash id) so the prefix cache can safely reuse
        an identical image's KV. Raises ValueError with a client-actionable
        message on failure.
        """
        parts = extract_media_parts(messages)
        if not parts:
            return None
        self._ensure_loaded()

        pixel_rows: list[torch.Tensor] = []
        grids: list[list[int]] = []
        for i, (kind, value) in enumerate(parts):
            frames = self._decode_frames(kind, value)
            pixels, grid = self._encode_frames(frames)
            t, h, w = grid
            # Client error, not a GPU OOM: the tower's attention/pos-embed transients
            # scale with the patch count, and this checkpoint's preprocessor_config
            # allows 16.7M-pixel images (65536 patches; whole-tower transient at that
            # size measured ~2.6 GiB fused / ~4.6 GiB on the chunked fallback).
            # 32768 patches (8.4 MP, ~2.6x vLLM's default pixel budget) keeps the
            # vision transient inside the engine's startup headroom. The budget
            # covers ALL frames of a video together (t counts toward it), which is
            # what keeps the eager vision tower inside the same ceiling.
            if t * h * w > 32_768:
                raise ValueError(
                    f"image {i} resized to {t*h*w} patches (grid {t}x{h}x{w}), over the "
                    "32768-patch engine budget; send a smaller image"
                )
            pixel_rows.append(pixels.reshape(-1, pixels.shape[-1]).float())
            grids.append([t, h, w])

        new_ids: list[torch.Tensor] = []
        types: list[torch.Tensor] = []
        cursor = 0
        for i, (t, h, w) in enumerate(grids):
            n = t * (h // self._merge) * (w // self._merge)
            if n <= 0 or h % self._merge or w % self._merge:
                raise ValueError(f"image {i} produced a degenerate grid {t},{h},{w} (merge={self._merge})")
            rest = input_ids[cursor:]
            # Both pad spellings count as the vision span (image_token_id for
            # image parts, video_token_id for videos); the expanded block is
            # written as image_token_id either way, so the model-side embed
            # mask (single-id) sees every media span.
            hit = (rest.eq(self._image_token_id) | rest.eq(self._video_token_id)).nonzero().flatten()
            if hit.numel() == 0:
                raise ValueError(
                    f"prompt has no image_pad placeholder for image {i}; the chat template "
                    "rendered no vision span"
                )
            end = cursor + int(hit[0].item())
            new_ids.append(input_ids[cursor:end])
            new_ids.append(torch.full((n,), self._image_token_id, dtype=input_ids.dtype))
            types.append(torch.zeros(end - cursor))
            types.append(torch.ones(n))
            cursor = end + 1  # just past the consumed single pad, so the next image finds ITS pad
        tail = input_ids[cursor:]
        extra = int((tail.eq(self._image_token_id) | tail.eq(self._video_token_id)).sum())
        if extra:
            raise ValueError(
                f"prompt has {extra} more image_pad token(s) than "
                "provided image(s); the template and the request disagree"
            )
        if new_ids:
            new_ids.append(tail)
            types.append(torch.zeros(tail.numel()))
        ids = torch.cat(new_ids) if new_ids else input_ids
        mm_types = torch.cat(types) if types else torch.zeros_like(input_ids)

        table = self._mrope_table(mm_types, grids)
        payload_path = _stage_payload(torch.cat(pixel_rows, 0), torch.tensor(grids, dtype=torch.int64))
        cache_key = _content_key(ids, self._image_token_id, payload_path)
        return ids, table.contiguous(), payload_path, cache_key

    def _mrope_table(
        self, types: torch.Tensor, grids: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        """HF get_rope_index (images and videos): a group scan over (text, image)
        runs. A text run advances all three channels by the token index. An image
        or video run with grid (t, h, w) starts at ``start`` and lays its
        ``t x (h/merge) x (w/merge)`` tokens row-major (t slowest, w fastest)
        with T = start + t_idx, H = start + h_idx, W = start + w_idx; the running
        position then advances by max(h, w)/merge -- the image's longest side,
        per HF -- so the prompt's mrope index stays <= its token count."""
        p = types.numel()
        table = torch.zeros(p, 3, dtype=torch.int32)
        idx = 0
        current_pos = 0
        grid_it = iter(grids)
        for modality, start_i, end_i in _runs(types):
            length = end_i - start_i
            if modality == 0:
                table[start_i:end_i] = torch.arange(
                    current_pos, current_pos + length, dtype=torch.int32
                )[:, None]
                current_pos += length
            else:
                t, h, w = next(grid_it)
                gh, gw = h // self._merge, w // self._merge
                start = current_pos
                j = torch.arange(length)
                ti = j // (gh * gw)
                hi = (j // gw) % gh
                wi = j % gw
                table[start_i:end_i, 0] = (start + ti).to(torch.int32)
                table[start_i:end_i, 1] = (start + hi).to(torch.int32)
                table[start_i:end_i, 2] = (start + wi).to(torch.int32)
                current_pos += max(gh, gw)
            idx += length
        if idx != p:  # pragma: no cover — _runs partitions the range exactly
            raise RuntimeError("mrope group scan did not cover the prompt")
        return table


def _runs(types: torch.Tensor) -> list[tuple[int, int, int]]:
    """Consecutive equal-modality runs as (value, start, end) over the token range."""
    flat = types.tolist()
    runs: list[tuple[int, int, int]] = []
    start = 0
    for i, v in enumerate(flat):
        if i + 1 < len(flat) and flat[i + 1] == v:
            continue
        runs.append((v, start, i + 1))
        start = i + 1
    return runs


# P4: the prefix cache keys on token ids, but image_pad ids are identical across images
# while their KV differs, so a raw-id match would serve the wrong image's KV. The payload
# .pt is already named by the sha1 of its pixels, so that digest is the natural content
# key: map it into a high int32 range far above any real token id (vocab ~150k) so a text
# token can never collide with a content id, and so two identical images always produce
# the SAME id (KV reuse) while different images produce different ones (no false match).
_CONTENT_ID_BASE = 1_000_000_000
_CONTENT_ID_SPAN = 1_000_000_000  # ids land in [1e9, 2e9): within int32 (2.147e9), >> vocab


def _content_id(payload_path: str) -> int:
    """One content id for THIS request's pixel payload, derived from its sha1 filename."""
    base = os.path.basename(payload_path)  # ft_mm_<sha1hex>.pt
    digest = base[len("ft_mm_") : -len(".pt")]
    return _CONTENT_ID_BASE + (int(digest, 16) % _CONTENT_ID_SPAN)


def _content_key(
    ids: torch.Tensor, image_token_id: int, payload_path: str
) -> torch.Tensor:
    """input_ids with every image_pad position replaced by the payload content id.

    This is the request's prefix-cache key: identical images -> identical key (the
    shared text plus the whole image span reuse cached KV); different images -> a
    different id at the span (no false match). The MODEL still sees the real input_ids
    (real pad ids) for attention/positions -- only the cache key changes.
    """
    key = ids.clone()
    key[ids == image_token_id] = _content_id(payload_path)
    return key


def _stage_payload(pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> str:
    """Write the payload to the OS temp dir, named by content hash (an identical
    image dedups end to end; the P4 radix key hashes the same bytes)."""
    body = pixel_values.numpy().tobytes() + grid_thw.numpy().tobytes()
    name = "ft_mm_" + hashlib.sha1(body).hexdigest()[:32] + ".pt"
    path = os.path.join(tempfile.gettempdir(), name)
    if not os.path.isfile(path):
        tmp = path + f".{os.getpid()}.tmp"
        torch.save({"pixel_values": pixel_values, "image_grid_thw": grid_thw}, tmp)
        os.replace(tmp, path)
    return path


def vision_geometry(tokenizer_path: str) -> dict[str, Any] | None:
    """(image_token_id, vision_config) for a multimodal checkpoint, or None.

    Read from the checkpoint files (no model build): the image-pad id from
    config.json (``image_token_id``) or the tokenizer's added tokens
    (``image_pad``), and the vision_config block for the spatial merge size.
    """
    try:
        with open(os.path.join(tokenizer_path, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return None
    text = cfg.get("text_config") or cfg
    image_token_id = cfg.get("image_token_id") or text.get("image_token_id")
    if image_token_id is None:
        try:
            with open(os.path.join(tokenizer_path, "tokenizer.json"), encoding="utf-8") as fh:
                tok = json.load(fh)
            for t in tok.get("added_tokens", []):
                if str(t.get("content", "")) == "image_pad":
                    image_token_id = t.get("id")
                    break
        except Exception:
            pass
    if image_token_id is None:
        return None
    video_token_id = cfg.get("video_token_id") or text.get("video_token_id") or image_token_id
    return {
        "image_token_id": int(image_token_id),
        "video_token_id": int(video_token_id),
        "vision_config": cfg.get("vision_config") or text.get("vision_config") or {},
    }


__all__ = [
    "MediaPipeline",
    "extract_image_urls",
    "extract_media_parts",
    "fetch_image",
    "fetch_video_frames",
    "vision_geometry",
]
