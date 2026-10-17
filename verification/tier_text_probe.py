"""Three-leg text-integrity probe for the ours tree (live serve).

Bisects "garbled output after turns" between the three ours-only layers:
  leg1 cold prompt A          -> baseline text (pure prefill path)
  leg2 repeat prompt A        -> hits the RAM tier; diff vs leg1 localizes tier corruption
  leg3 different prompt B     -> must be independent of A (no cross-entry bleed)

temperature=0 makes leg1/leg2 expected-identical; the FIRST MISMATCH OFFSET is the
diagnostic: offset ~0 -> restore path itself; offset ~cache-boundary -> wrong
prefix boundary parked; leg3-only garble -> slot reuse/aliasing.

  python tier_text_probe.py [base_url] [ctx_tokens]
"""
import difflib
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 2000


def one(prompt: str, max_tokens: int = 192):
    payload = {"model": "qwen",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    body = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
    msg = body.get("choices", [{}])[0].get("message", {})
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    u = body.get("usage", {})
    return text, u


def text_a(ntok):
    return ("Explain how a shared cache arena keeps one side left-anchored and the other "
            "right-anchored so the boundary slides without reallocating. " * max(1, ntok // 24)
            + "\nSummarize the mechanism in two sentences.")


def text_b(ntok):
    return ("Describe in plain words why a scheduler applies pool rebalancing only when "
            "the queue is quiet. " * max(1, ntok // 20)
            + "\nGive the answer in two sentences.")


def first_diff(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            return max(i1, j1)
    return min(len(a), len(b))


def main() -> int:
    pa, pb = text_a(CTX), text_b(max(512, CTX // 4))

    a1, u1 = one(pa)
    a2, u2 = one(pa)
    b1, u3 = one(pb)

    ct1 = u1.get("prompt_tokens_details", {}).get("cached_tokens")
    ct2 = u2.get("prompt_tokens_details", {}).get("cached_tokens")
    print(f"leg1 prompt={u1.get('prompt_tokens')}t len={len(a1)} cached={ct1}")
    print(f"leg2 prompt={u2.get('prompt_tokens')}t len={len(a2)} cached={ct2}")
    print(f"leg3 prompt={u3.get('prompt_tokens')}t len={len(b1)} cached={u3.get('prompt_tokens_details', {}).get('cached_tokens')}")

    fails = []
    if min(len(a1), len(a2), len(b1)) == 0:
        fails.append("empty response on some leg")
    if a1 == a2:
        print(f"leg2_vs_leg1: identical (good)")
    else:
        off = first_diff(a1, a2)
        ratio = difflib.SequenceMatcher(a=a1, b=a2, autojunk=False).ratio()
        print(f"leg2_vs_leg1: DIVERGE first_diff_char={off} similarity={ratio:.2f}")
        print(f"  leg1[{max(0,off-40)}:{off+80}]: {a1[max(0,off-40):off+80]!r}")
        print(f"  leg2[{max(0,off-40)}:{off+80}]: {a2[max(0,off-40):off+80]!r}")
        if ratio < 0.6:
            fails.append("repeat-turn text diverged strongly (tier restore suspect)")
    if ct2 and ct1 is not None and ct2 == 0 and u2.get("prompt_tokens", 0) > 500:
        fails.append("big repeat still cached_tokens=0 (tier miss)")

    print()
    if fails:
        print(f"RESULT: FAIL {fails}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
