"""Live-serve smoke for the video path (runs against the running serve on :8001).

  python -X utf8 verification/serve_video_smoke.py
"""

import base64
import io
import json
import os
import urllib.request

BASE = os.environ.get("FREETOKEN_TEST_BASE", "http://127.0.0.1:8001")
MODEL = "Qwen3.8-Flash-Next-ABLITERATED-NVFP4"
MP4 = os.environ.get("FREETOKEN_TEST_MP4", "sample.mp4")


def chat(content, max_tokens=1024):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        print(f"[http {exc.code}] {exc.read().decode()[:600]}")
        raise
    return out["choices"][0]["message"]["content"]


def png_uri(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    from PIL import Image

    # three distinct synthetic frames -> PNG data URIs (frame-list video form)
    frames = []
    for i, col in enumerate([(220, 40, 40), (40, 160, 60), (40, 80, 220)]):
        frames.append(png_uri(Image.new("RGB", (96, 64), col)))

    text = chat([{"type": "text", "text": "Reply with exactly: pong"}], 24)
    print(f"[1 text-only]    -> {text!r}")

    vresp = chat(
        [{"type": "text", "text": "How many colors of background appear across these frames? Name them."}]
        + [{"type": "video", "video": frames}],
    )
    print(f"[2 video frames] -> {vresp!r}")

    # animated GIF as a single file reference
    gif_path = "_smoke_frames.gif"
    ims = [Image.new("RGB", (96, 64), c) for c in [(220, 40, 40), (40, 80, 220)]]
    ims[0].save(gif_path, save_all=True, append_images=ims[1:], duration=100, loop=0)
    gresp = chat(
        [{"type": "text", "text": "Which background colors appear in this animation?"},
         {"type": "video", "video": gif_path}],
    )
    print(f"[3 gif file]     -> {gresp!r}")

    mixed = chat(
        [{"type": "text", "text": "Is the still image's color also present in the animation? Which?"},
         {"type": "image_url", "image_url": {"url": frames[1]}},
         {"type": "video", "video": gif_path}],
    )
    print(f"[4 image+video]  -> {mixed!r}")

    mp4resp = chat(
        [{"type": "text", "text": "One sentence: what is in this video?"},
         {"type": "video", "video": MP4}],
    )
    print(f"[5 mp4 path]     -> {mp4resp!r}")

    ok = all(isinstance(t, str) and len(t.strip()) > 0 for t in (text, vresp, gresp, mixed))
    print("SMOKE:", "OK" if ok else "EMPTY RESPONSES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
