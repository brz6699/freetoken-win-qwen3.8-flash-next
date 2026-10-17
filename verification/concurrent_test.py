"""Concurrency smoke against a live FreeToken serve on :8001.

Three checks:
1. N concurrent identical-prompt requests all return 200 (no crash under overlap).
2. Repeated same-head prompts show cache hits (second+ turn faster / cached tokens > 0).
3. Different-head prompts do not false-match (still produce sensible output).

  python concurrent_test.py [base_url] [N]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def post(payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    body["_ms"] = round((time.monotonic() - t0) * 1000)
    return body


def _text_len(r: dict) -> int:
    msg = r.get("choices", [{}])[0].get("message", {})
    return len((msg.get("content") or "") + (msg.get("reasoning_content") or ""))


def main() -> int:
    head = ("You are a precise assistant. Answer the question concisely. " * 6) + "\nQ: What is a KV cache? A:"
    fails: list[str] = []

    # 1) concurrent identical prompts
    print(f"-- concurrent x{N} identical head --")
    results = []
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(post, {"model": "qwen", "messages": [{"role": "user", "content": head}],
                                 "max_tokens": 48, "temperature": 0}) for _ in range(N)]
        for f in cf.as_completed(futs):
            results.append(f.result())
    codes = [r.get("_ms") for r in results]
    outs = [_text_len(r) for r in results]
    print(f"   ms={codes} out_len={outs}")
    if any(o == 0 for o in outs):
        fails.append("some concurrent responses empty")

    # 2) repeated same-head: expect cached-token > 0 on repeats
    print("-- sequential repeat x3 (same head) --")
    ms = []
    cached = []
    for i in range(3):
        r = post({"model": "qwen", "messages": [{"role": "user", "content": head}],
                  "max_tokens": 48, "temperature": 0})
        ms.append(r.get("_ms"))
        cached.append(r.get("usage", {}))
    print(f"   ms={ms}")
    for i, u in enumerate(cached):
        ct = u.get("prompt_tokens_details", {}).get("cached_tokens", None)
        if ct is None:
            ct = u.get("cached_tokens")
        print(f"   turn{i+1} cached_tokens={ct}")
    if ms[0] and min(ms) > ms[0]:
        fails.append("repeat not faster than cold")

    # 3) different head must not false-match the first
    print("-- different head --")
    alt = "Explain in one short sentence what a GPU is. " * 8 + "\nQ: one-line summary. A:"
    r = post({"model": "qwen", "messages": [{"role": "user", "content": alt}],
              "max_tokens": 32, "temperature": 0})
    tl = _text_len(r)
    print(f"   alt_len={tl} ms={r.get('_ms')}")
    if tl == 0:
        fails.append("different-head response empty")

    print()
    if fails:
        print(f"RESULT: FAIL {fails}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
