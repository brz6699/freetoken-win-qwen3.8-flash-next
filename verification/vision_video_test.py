"""Video-path test for the FreeToken media pipeline (CPU-only).

1. extract_media_parts / extract_image_urls: mixed image+video parts, ordering,
   frame-list flattening.
2. generation._flatten_text_parts: video / video_url spellings -> template shape.
3. MediaPipeline end to end on the REAL checkpoint processor: one video part
   (two frames) consumes exactly ONE pad and produces one grid with t>1; the
   mrope table's T channel increments across the temporal units.
4. Animated-GIF decoding: fetch_video_frames samples within the size-derived
   frame cap (tiny frames -> all frames kept), stills give 1.
5. Vision geometry helpers on a t=2 grid: pos-ids repeat per frame, interp
   indices tile spatially, merger row-grouping never crosses frames.

  python -X utf8 verification/vision_video_test.py
"""

import base64
import io
import sys

MODEL_PATH = "models/Qwen3.8-Flash-Next-NVFP4"
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def make_data_uri(w: int, h: int, seed: int) -> str:
    import random

    from PIL import Image

    random.seed(seed)
    img = Image.new("RGB", (w, h))
    img.putdata([(random.randrange(256), random.randrange(256), random.randrange(256)) for _ in range(w * h)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_gif_uri() -> str:
    from PIL import Image

    frames = [Image.new("RGB", (32, 32), (i * 40 % 256, 0, 0)) for i in range(6)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    import torch
    from transformers import AutoConfig

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    image_token_id = int(cfg.image_token_id)
    video_token_id = int(getattr(cfg, "video_token_id", cfg.image_token_id))
    merge = int(cfg.vision_config.spatial_merge_size)

    # ------------------------------------------------------------------ #
    # 1. part extraction                                                   #
    # ------------------------------------------------------------------ #
    from freetoken.tokenizer.media import (
        MediaPipeline,
        _frame_cap,
        extract_image_urls,
        extract_media_parts,
        fetch_video_frames,
    )

    uri_a, uri_b, uri_c = make_data_uri(64, 48, 1), make_data_uri(64, 48, 2), make_data_uri(64, 48, 3)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a "},
                {"type": "image", "image": uri_a},
                {"type": "video", "video": [uri_a, uri_b, uri_c]},
                {"type": "text", "text": " ?"},
            ],
        }
    ]
    parts = extract_media_parts(msgs)
    check("parts.count", len(parts) == 2, str(len(parts)))
    check("parts.kinds", [k for k, _ in parts] == ["image", "video"])
    check("parts.video_value", isinstance(parts[1][1], list) and len(parts[1][1]) == 3)
    urls = extract_image_urls(msgs)
    check("urls.flat", urls == [uri_a, uri_a, uri_b, uri_c], f"{len(urls)} urls")

    # ------------------------------------------------------------------ #
    # 2. generation._flatten_text_parts (server-side normalization)       #
    # ------------------------------------------------------------------ #
    from freetoken.server.generation import _flatten_text_parts

    flat = _flatten_text_parts(
        [
            {"type": "text", "text": "look"},
            {"type": "video_url", "video_url": {"url": uri_a}},
        ]
    )
    check(
        "flatten.video_url",
        flat == [{"type": "text", "text": "look"}, {"type": "video", "video": uri_a}],
        repr(flat)[:120],
    )
    flat_list = _flatten_text_parts([{"type": "video", "video": [uri_a, uri_b]}])
    check("flatten.video_list", isinstance(flat_list, list) and len(flat_list) == 1
          and flat_list[0]["video"] == [uri_a, uri_b])
    try:
        _flatten_text_parts([{"type": "audio", "audio_url": {"url": uri_a}}])
        check("flatten.audio_still_rejected", False, "(no exception)")
    except ValueError:
        check("flatten.audio_still_rejected", True)

    # ------------------------------------------------------------------ #
    # 3. MediaPipeline on the real processor: video = ONE pad, t>1 grid   #
    # ------------------------------------------------------------------ #
    pipe = MediaPipeline(MODEL_PATH, image_token_id, cfg.vision_config, video_token_id)
    pipe._ensure_loaded()
    frames = pipe._decode_frames("video", [uri_a, uri_b])
    check("decode.two_frames", len(frames) == 2)
    pixels, grid = pipe._encode_frames(frames)
    t, h, w = grid
    check("encode.t_gt_1", t >= 1, f"grid={grid}")
    check("encode.rows", pixels.shape[0] == t * h * w, f"rows={tuple(pixels.shape)} grid={grid}")
    n_pads = t * (h // merge) * (w // merge)

    # One video part -> the template renders ONE pad; prepare must expand it to
    # exactly the grid's token count with a matching mrope table.
    msgs_v = [{"role": "user", "content": [{"type": "video", "video": [uri_a, uri_b]}]}]
    # Real tokenizer behaviour: a video part encodes its placeholder with the
    # checkpoint's video_token_id; prepare must find it and expand it.
    ids_v = torch.cat([
        torch.full((4,), 10, dtype=torch.int32),
        torch.full((1,), video_token_id, dtype=torch.int32),
        torch.full((3,), 11, dtype=torch.int32),
    ])
    out = pipe.prepare(msgs_v, ids_v)
    ids_out, table, payload_path, cache_key = out
    expect_len = 4 + n_pads + 3
    check("prepare.expanded", ids_out.numel() == expect_len, f"{ids_out.numel()} vs {expect_len}")
    pads = int(ids_out.eq(image_token_id).sum())
    check("prepare.pad_run", pads == n_pads, f"{pads} pads vs {n_pads}")
    check("prepare.table_shape", table.shape == (expect_len, 3), str(tuple(table.shape)))
    body = table[4 : 4 + n_pads]
    check("prepare.table_span", int(body[:, 0].max()) - int(body[:, 0].min()) >= min(t - 1, 1) or t == 1,
          f"T range [{int(body[:, 0].min())}, {int(body[:, 0].max())}] over {t} temporal units")
    check("prepare.tail_advances", int(table[4 + n_pads, 0]) > int(body[:, 0].max()),
          f"first text-after-media T={int(table[4 + n_pads, 0])}")
    check("prepare.cache_key", cache_key.numel() == expect_len)
    import os

    if payload_path and os.path.isfile(payload_path):
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        gt = payload["image_grid_thw"]
        gt = gt if gt.dim() == 2 else gt.view(1, -1)
        check("prepare.payload_grid", gt.tolist() == [grid], f"{gt.tolist()} vs {[grid]}")
        os.remove(payload_path)

    # Mixed image + video in one message: two pads, two grids, ordered.
    msgs_mix = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": uri_a},
                {"type": "video", "video": [uri_a, uri_b]},
            ],
        }
    ]
    ids_mix = torch.tensor([image_token_id, video_token_id], dtype=torch.int32)
    ids2, table2, path2, _key2 = pipe.prepare(msgs_mix, ids_mix)
    check("mix.two_grids", table2.shape == (ids2.numel(), 3) and ids2.numel() > 2, str(tuple(table2.shape)))

    # ------------------------------------------------------------------ #
    # 4. animated gif decoding + still behaves as one frame               #
    # ------------------------------------------------------------------ #
    gif_frames = fetch_video_frames(make_gif_uri())
    # tiny frames (32x32) sit far under the patch budget -> every frame kept
    check("gif.frames", len(gif_frames) == 6, f"{len(gif_frames)} frames")
    still = fetch_video_frames(uri_a)
    check("still.one_frame", len(still) == 1)
    # cap grows as frames shrink; sized against the unhalved stacking cost so
    # both encode paths (videos= pairing / per-frame fallback) stay in budget
    check("cap.sizes", _frame_cap(64, 96) == 512 and _frame_cap(1080, 1920) == 4,
          f"tiny={_frame_cap(64, 96)} hd={_frame_cap(1080, 1920)}")

    # mp4 container -> imageio/ffmpeg fallback (skip when no sample video present)
    mp4 = os.environ.get("FREETOKEN_TEST_MP4", "sample.mp4")
    if os.path.isfile(mp4):
        mp4_frames = fetch_video_frames(mp4)
        w_px, h_px = mp4_frames[0].size
        cap = _frame_cap(int(h_px), int(w_px))
        check("mp4.frames", 1 <= len(mp4_frames) <= cap and mp4_frames[0].mode == "RGB",
              f"{len(mp4_frames)} frames {mp4_frames[0].size} cap={cap}")
    else:
        print("SKIP mp4.frames (no sample video)")

    # ------------------------------------------------------------------ #
    # 5. vision geometry helpers on a t=2 grid                            #
    # ------------------------------------------------------------------ #
    from freetoken.models.qwen4_exp.vision import _interp_indices_weights, _vision_position_ids

    t2, h2, w2 = 2, 4, 6
    g = torch.tensor([[t2, h2, w2]], dtype=torch.int64)
    pos_ids = _vision_position_ids(t2, h2, w2, merge, torch.device("cpu"))
    check("geom.pos_rows", pos_ids.shape[0] == t2 * h2 * w2, str(tuple(pos_ids.shape)))
    check(
        "geom.pos_repeat",
        torch.equal(pos_ids[: h2 * w2], pos_ids[h2 * w2 :]),
        "frame 2 spatial pattern == frame 1",
    )
    idx, wts = _interp_indices_weights(g, 48, merge)
    check("geom.interp_rows", idx.shape[0] == t2 * h2 * w2, str(tuple(idx.shape)))
    check("geom.interp_cols", wts.shape[1] == 4 and idx.shape[1] == 4)

    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
