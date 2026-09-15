"""Vision P3 media-pipeline test (CPU-only, runs beside the live serve).

1. mrope table: engine MediaPipeline._mrope_table vs the HF oracle
   Qwen4ExpModel.get_rope_index (meta-device model, real checkpoint config)
   -- text runs, two images, post-image position jumps, base/delta rule.
2. Worker path on the REAL checkpoint: chat-template render with image content
   parts (data: URIs) -> TokenizeManager.tokenize -> expanded ids, mrope table,
   staged .pt payload; then the UserMsg serializer round trip (the exact online
   transport, incl. the flat [P*3] reshape the scheduler performs).
3. QSA indexer mrope kernel: qsa_index_norm_rope with pos3/mrope_sel (small
   GPU scratch, ~50 MB) vs a torch reference on the same cos/sin table.

  python -X utf8 verification/vision_p3_media_test.py
"""

import base64
import io
import os
import sys

MODEL_PATH = os.environ.get("FREETOKEN_TEST_MODEL",
                            "models/Qwen3.8-Flash-Next-NVFP4")
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


def pil_from_uri(uri: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))


def main() -> int:
    import torch
    from transformers import AutoConfig

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    tc = cfg.text_config if getattr(cfg, "text_config", None) else cfg
    image_token_id = int(cfg.image_token_id)
    merge = int(cfg.vision_config.spatial_merge_size)
    rope_theta = float(getattr(tc, "rope_theta", 10000.0))
    print(f"image_token_id={image_token_id} merge={merge} rope_theta={rope_theta}")

    # ------------------------------------------------------------------ #
    # 1. mrope table vs HF oracle                                         #
    # ------------------------------------------------------------------ #
    from freetoken.tokenizer.media import MediaPipeline

    pipe = MediaPipeline(MODEL_PATH, image_token_id, cfg.vision_config)

    # seg: 50 text | 24 img(1,8,12) | 20 text | 32 img(1,16,8) | 10 text
    seg = [50, 24, 20, 32, 10]
    P = sum(seg)
    types = torch.cat(
        [torch.zeros(seg[0]), torch.ones(seg[1]), torch.zeros(seg[2]),
         torch.ones(seg[3]), torch.zeros(seg[4])]
    )
    grids = [[1, 8, 12], [1, 16, 8]]
    mine = pipe._mrope_table(types, grids)
    check("mrope.shape", mine.shape == (P, 3) and mine.dtype == torch.int32, str(tuple(mine.shape)))
    check(
        "mrope.text_head",
        torch.equal(mine[:50, 0], torch.arange(50, dtype=torch.int32))
        and torch.equal(mine[:50, 1], mine[:50, 0]) and torch.equal(mine[:50, 2], mine[:50, 0]),
    )
    j = torch.arange(24)
    b1 = mine[50:74]
    check("mrope.img1.T", torch.equal(b1[:, 0], torch.full((24,), 50, dtype=torch.int32)))
    check("mrope.img1.H", torch.equal(b1[:, 1], (50 + (j // 6) % 4).to(torch.int32)))
    check("mrope.img1.W", torch.equal(b1[:, 2], (50 + j % 6).to(torch.int32)))
    check("mrope.jump_after_img1", int(mine[74, 0]) == 56,
          f"post-image text mrope pos = {int(mine[74, 0])} (token index 74)")
    j2 = torch.arange(32)
    b2 = mine[94:126]
    check("mrope.img2.T", torch.equal(b2[:, 0], torch.full((32,), 76, dtype=torch.int32)))
    check("mrope.img2.H", torch.equal(b2[:, 1], (76 + (j2 // 4) % 8).to(torch.int32)))
    check("mrope.img2.W", torch.equal(b2[:, 2], (76 + j2 % 4).to(torch.int32)))
    check("mrope.tail", int(mine[126, 0]) == 84, f"got {int(mine[126, 0])}")

    hf_ran = False
    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

        dummy_ids = torch.zeros(P, dtype=torch.int64)
        dummy_ids[50:74] = image_token_id
        dummy_ids[96:128] = image_token_id
        with torch.device("meta"):
            hf_model = Qwen4ExpModel(cfg)
        pos_ids, deltas = hf_model.get_rope_index(
            dummy_ids.unsqueeze(0),
            types.unsqueeze(0).int(),
            image_grid_thw=torch.tensor(grids, dtype=torch.int64),
        )
        hf_table = pos_ids[:, 0].cpu().to(torch.int64)  # [3, P] (T/H/W rows, batch 0)
        mine_t = mine.to(torch.int64).t().contiguous()
        if torch.equal(hf_table, mine_t):
            check("mrope.hf_equal", True, "(3 x P table bitwise equal)")
        else:
            diff = (hf_table != mine_t).nonzero().flatten()
            first = diff[:5].tolist()
            shown = [(int(d), hf_table[0, d].item(), hf_table[1, d].item(), hf_table[2, d].item(),
                      mine_t[0, d].item(), mine_t[1, d].item(), mine_t[2, d].item()) for d in first]
            check("mrope.hf_equal", False, f"{diff.numel()} diffs; first (tok, hf t/h/w, mine t/h/w): {shown}")
        delta = int(deltas.flatten()[0])
        base = int(mine.max().item()) + 1
        check("mrope.base_rule", base - P == delta, f"base={base} len={P} hf_delta={delta}")
        hf_ran = True
    except Exception as exc:  # noqa: BLE001
        check("mrope.hf_oracle", False, f"oracle unavailable: {exc!r}")
    if hf_ran:
        print("  (HF oracle ran on meta device)")

    # ------------------------------------------------------------------ #
    # 2. worker path: real tokenizer + processor + serializer round trip  #
    # ------------------------------------------------------------------ #
    from freetoken.core import SamplingParams
    from freetoken.message import TokenizeMsg, UserMsg
    from freetoken.message.backend import deserialize_type
    from freetoken.message.utils import serialize_type
    from freetoken.tokenizer.media import extract_image_urls
    from freetoken.tokenizer.tokenize import TokenizeManager

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    image_pad_str = tokenizer.decode([image_token_id])
    print(f"image_pad token: {image_pad_str!r}")

    uri_a = make_data_uri(160, 120, seed=1)
    uri_b = make_data_uri(200, 100, seed=2)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in "},
                {"type": "image", "image": uri_a},
                {"type": "text", "text": " and "},
                {"type": "image", "image": uri_b},
                {"type": "text", "text": " together?"},
            ],
        }
    ]
    check("media.extract", extract_image_urls(messages) == [uri_a, uri_b])

    mgr = TokenizeManager(tokenizer)
    check("media.vision_supported", bool(mgr._vision_supported), "(checkpoint probe)")

    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams())
    prompt = mgr.render_prompt(msg)
    n_pads = prompt.count(image_pad_str)
    check("template.pads", n_pads == 2, f"rendered prompt has {n_pads} pads (want 2), len={len(prompt)}")

    res = mgr.tokenize([msg])[0]
    ids = res.input_ids

    pipe._ensure_loaded()  # part-1 pipeline never ran the processor; load it for the cross-check
    out_a = pipe._processor(images=pil_from_uri(uri_a), return_tensors="pt")
    out_b = pipe._processor(images=pil_from_uri(uri_b), return_tensors="pt")
    ga = [int(v) for v in out_a["image_grid_thw"].flatten().tolist()]
    gb = [int(v) for v in out_b["image_grid_thw"].flatten().tolist()]
    na = ga[1] // merge * ga[2] // merge * ga[0]
    nb = gb[1] // merge * gb[2] // merge * gb[0]
    print(f"grid_a={ga} n_a={na}  grid_b={gb} n_b={nb}")

    pad_mask = ids.eq(image_token_id)
    run_lengths, prev = [], False
    for v in pad_mask.tolist():
        if v and not prev:
            run_lengths.append(1)
        elif v:
            run_lengths[-1] += 1
        prev = v
    check("worker.two_blocks", len(run_lengths) == 2, f"runs={run_lengths}")
    check("worker.block_sizes", run_lengths == [na, nb], f"got {run_lengths}, want [{na}, {nb}]")
    check(
        "worker.mrope_shape",
        res.mm_mrope is not None and res.mm_mrope.shape == (len(ids), 3),
        str(tuple(res.mm_mrope.shape) if res.mm_mrope is not None else None),
    )
    check("worker.payload", bool(res.mm_data_path) and os.path.isfile(res.mm_data_path),
          res.mm_data_path or "(none)")

    if res.mm_data_path:
        payload = torch.load(res.mm_data_path, map_location="cpu", weights_only=True)
        pv, gt = payload["pixel_values"], payload["image_grid_thw"]
        rows_a = ga[0] * ga[1] * ga[2]
        rows_b = gb[0] * gb[1] * gb[2]
        check("payload.grid", gt.tolist() == [ga, gb], str(gt.tolist()))
        check("payload.rows", pv.shape[0] == rows_a + rows_b, f"{pv.shape[0]} vs {rows_a + rows_b}")

    # --- mrope table of the REAL worker output (expanded ids + processor grids) vs HF oracle ---
    table = res.mm_mrope
    first_img = int(pad_mask.nonzero().flatten()[0])
    prefix = torch.arange(first_img, dtype=torch.int32)
    check(
        "worker.table_prefix",
        bool((table[:first_img] == prefix[:, None]).all()),
        "(3 channels = token index on every pre-image text row)",
    )
    hf_worker_ran = False
    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

        with torch.device("meta"):
            hf_w = Qwen4ExpModel(cfg)
        wids = ids.to(torch.int64).unsqueeze(0)
        wtypes = pad_mask.to(torch.int32).unsqueeze(0)
        wpos, wdelta = hf_w.get_rope_index(
            wids, wtypes, image_grid_thw=torch.tensor([ga, gb], dtype=torch.int64)
        )
        hf_w_table = wpos[:, 0].to(torch.int64)
        mine_w = table.to(torch.int64).t().contiguous()
        if torch.equal(hf_w_table, mine_w):
            check("worker.hf_table_equal", True, "(3 x P table bitwise equal to HF on the real prompt)")
        else:
            d = (hf_w_table != mine_w).nonzero().flatten()
            check("worker.hf_table_equal", False, f"{d.numel()} cell diffs")
        base = int(table.max()) + 1
        check(
            "worker.hf_base_rule",
            base - len(ids) == int(wdelta.flatten()[0]),
            f"base={base} len={len(ids)} hf_delta={int(wdelta.flatten()[0])}",
        )
        hf_worker_ran = True
    except Exception as exc:  # noqa: BLE001
        check("worker.hf_oracle", False, f"oracle unavailable: {exc!r}")
    if hf_worker_ran:
        print("  (HF oracle ran on the real worker prompt)")
    check("worker.table_bounded", int(table.max()) < len(ids), f"max={int(table.max())} < {len(ids)}")

    # serializer round trip (the online wire): flat [P*3], scheduler reshapes
    umsg = UserMsg(
        uid=1,
        input_ids=ids,
        sampling_params=SamplingParams(),
        mm_mrope=res.mm_mrope.contiguous().view(-1).to(torch.int32),
        mm_data_path=res.mm_data_path,
    )
    blob = serialize_type(umsg)
    back = deserialize_type(globals() | {"UserMsg": UserMsg, "SamplingParams": SamplingParams}, blob)
    check("wire.roundtrip_type", isinstance(back, UserMsg))
    check("wire.ids", torch.equal(back.input_ids, ids))
    check(
        "wire.mrope_flat",
        back.mm_mrope is not None and back.mm_mrope.dim() == 1 and back.mm_mrope.shape[0] == len(ids) * 3,
        str(tuple(back.mm_mrope.shape) if back.mm_mrope is not None else None),
    )
    check("wire.path", back.mm_data_path == res.mm_data_path)
    if back.mm_mrope is not None and back.mm_mrope.dim() == 1:
        back.mm_mrope = back.mm_mrope.view(-1, 3)
        check("wire.reshape", torch.equal(back.mm_mrope, res.mm_mrope))
    os.remove(res.mm_data_path)

    # ------------------------------------------------------------------ #
    # 2b. patch-budget guard (stubbed processor; real images >8.4 MP would #
    #     need GB-scale PNGs). The over-limit path must raise the           #
    #     client-actionable ValueError naming the image index — a missing   #
    #     enumerate() once made it an UnboundLocalError.                    #
    # ------------------------------------------------------------------ #
    class _StubProcessor:
        def __init__(self, t, h, w):
            self._g = (t, h, w)

        def __call__(self, images=None, return_tensors="pt"):
            t, h, w = self._g
            return {
                "pixel_values": torch.zeros(t * h * w, 3 * 2 * 16 * 16),
                "image_grid_thw": torch.tensor([t, h, w]),
            }

    uri_g = make_data_uri(16, 16, seed=9)
    msgs_g = [{"role": "user", "content": [{"type": "image", "image": uri_g}]}]
    ids_g = torch.cat([
        torch.full((7,), 10, dtype=torch.int64),
        torch.full((1,), image_token_id, dtype=torch.int64),
        torch.full((7,), 11, dtype=torch.int64),
    ])
    pipe_at = MediaPipeline(MODEL_PATH, image_token_id, cfg.vision_config)
    pipe_at._processor = _StubProcessor(1, 128, 256)  # exactly 32768 = at budget
    res_at = pipe_at.prepare(msgs_g, ids_g)
    check(
        "guard.at_budget_passes",
        res_at is not None and res_at[0].numel() == 14 + 8192,
        f"ids={res_at[0].numel()} (14 text + 8192 pads)" if res_at else "(rejected)",
    )
    if res_at and res_at[2] and os.path.isfile(res_at[2]):
        os.remove(res_at[2])
    pipe_over = MediaPipeline(MODEL_PATH, image_token_id, cfg.vision_config)
    pipe_over._processor = _StubProcessor(1, 129, 256)  # 33024 > 32768
    try:
        pipe_over.prepare(msgs_g, ids_g)
        check("guard.oversized_raises", False, "(no exception raised)")
    except ValueError as exc:
        check("guard.oversized_raises", "image 0" in str(exc) and "32768" in str(exc),
              repr(str(exc)[:120]))
    except Exception as exc:  # noqa: BLE001
        check("guard.oversized_raises", False,
              f"wrong type {type(exc).__name__}: {exc!r}")

    # ------------------------------------------------------------------ #
    # 3. QSA indexer mrope kernel (small GPU scratch)                      #
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        try:
            from freetoken.kernel.triton.qsa.compress import qsa_index_norm_rope
            from freetoken.layers.rotary import get_rope

            dev = "cuda"
            R, D, HALF = 4, 128, 32  # 2 tokens x 2 heads, index head dim 128, rotary 64
            torch.manual_seed(1)
            x = torch.randn(R, D, device=dev, dtype=torch.bfloat16)
            weight = torch.randn(D, device=dev, dtype=torch.bfloat16) * 0.1
            positions = torch.tensor([5, 5, 9, 40], dtype=torch.int32, device=dev)
            out = torch.empty_like(x)
            rope = get_rope(head_dim=128, rotary_dim=64, max_position=8192, base=rope_theta, rope_scaling=None)
            cos_sin = rope._cos_sin_cache.to(dev)
            # Distinct t/h/w per token so the per-frequency channel selection (sel 0/1/2)
            # actually reads three different cos/sin rows (all-equal triples would mask a bug).
            pos3 = torch.tensor([[5, 10, 20], [30, 40, 50]], dtype=torch.int32, device=dev)
            sel = torch.zeros(HALF, dtype=torch.int64, device=dev)
            sel[1::3] = 1
            sel[2::3] = 2
            qsa_index_norm_rope(x, positions, cos_sin, weight, 1e-6, out, heads=2, pos3=pos3, mrope_sel=sel)
            xf = x.float()
            rrms = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / D + 1e-6)
            y = xf * rrms * (weight.float() + 1.0)
            ok = True
            for tok in range(2):
                for h in range(2):
                    row = tok * 2 + h
                    triple = pos3[tok].tolist()
                    cos = torch.tensor([float(cos_sin[triple[int(sel[p])], p]) for p in range(HALF)], device=dev)
                    sin = torch.tensor([float(cos_sin[triple[int(sel[p])], HALF + p]) for p in range(HALF)], device=dev)
                    a, b = y[row, :HALF], y[row, HALF : 2 * HALF]
                    ref = torch.cat([a * cos - b * sin, b * cos + a * sin, y[row, 2 * HALF :]]).to(torch.bfloat16)
                    md = (out[row].float() - ref.float()).abs().max().item()
                    if md > 2e-2:
                        ok = False
                        print(f"  row {row} mismatch: max diff {md:.4f}")
            check("qsa.mrope_kernel", ok, "(2 tokens x 2 heads, sel 0/1/2 cycling)")
        except Exception as exc:  # noqa: BLE001
            check("qsa.mrope_kernel", False, f"unexpected: {exc!r}")
    else:
        print("SKIP qsa.mrope_kernel (no CUDA)")

    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
