"""Concurrency throughput-scaling + long-context stress against live :8001.

speed: same short prompt at bs1 / bs2 / bs4 -> per-request latency and aggregate
       decode throughput, so batch scaling is visible.
long : repeat-vs-first latency on a LONG prompt (cache reuse under load), plus a
       concurrent burst of long prompts.

Run several rounds for stability.

  python concurrent_stress.py [base_url] [rounds] [ctx_tokens]
"""
import json
import sys
import time
import urllib.request
import concurrent.futures as cf

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 8000


def one(prompt: str, max_tokens: int):
    payload = {"model": "qwen",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    body = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
    ms = int((time.monotonic() - t0) * 1000)
    msg = body.get("choices", [{}])[0].get("message", {})
    out_text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    u = body.get("usage", {})
    ctok = u.get("completion_tokens") or len(out_text) // 4 or len(out_text)
    return ms, ctok, u.get("prompt_tokens", 0)


def conc(n: int, prompt: str, max_tokens: int):
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in cf.as_completed(
            [ex.submit(one, prompt, max_tokens) for _ in range(n)])]


def small_prompt(ntok=300):
    return ("Summarize in one line what caching does here. ") * max(1, ntok // 9)


def long_prompt(ntok):
    return ("The cache arena keeps KV left-anchored and MoE slots right-anchored so the "
            "boundary slides without reallocating; the next turn reuses the parked rows. ") * max(1, ntok // 22)


def run_speed(rounds: int) -> None:
    print("== SPEED scaling ==")
    p = small_prompt(300)
    for n in (1, 2, 4):
        lat, walls, toks, ptok = [], [], 0, 0
        for _ in range(rounds):
            t0 = time.monotonic()
            r = conc(n, p, 32)
            walls.append(time.monotonic() - t0)
            lat += [x[0] for x in r]
            toks += sum(x[1] for x in r)
            ptok = r[0][2]
        avg_lat = sum(lat) / len(lat)
        agg = toks / sum(walls)
        reqs = len(lat) / sum(walls)
        print(f"  bs={n} prompt~{ptok}t avg_lat={avg_lat:.0f}ms agg_decode={agg:.1f}tok/s req_rate={reqs:.1f}req/s")


def run_long(rounds: int, ctx: int) -> None:
    print(f"== LONG stress (ctx~{ctx}t, cold-vs-repeat) ==")
    p = long_prompt(ctx)
    firsts, reps, ptok = [], [], 0
    for _ in range(rounds):
        f_ms, _, ptok = one(p, 16)
        firsts.append(f_ms)
        rr = [one(p, 16)[0] for _ in range(3)]
        reps.append(sum(rr) / len(rr))
    print(f"  prompt_tokens~{ptok}")
    print(f"  first_ms={firsts}")
    print(f"  repeat_avg_ms={[round(x) for x in reps]}")
    gain = firsts[0] / reps[-1] if reps[-1] else 0
    print(f"  repeat_speedup_x={gain:.2f}")
    t0 = time.monotonic()
    r = conc(4, long_prompt(ctx // 4), 24)
    wall = time.monotonic() - t0
    toks = sum(x[1] for x in r)
    print(f"  burst x4(~{ctx//4}t) wall={wall:.2f}s agg={toks/wall:.1f}tok/s per_req_ms={[x[0] for x in r]}")


def main() -> int:
    for rnd in range(ROUNDS):
        print(f"\n########## ROUND {rnd+1}/{ROUNDS} ##########")
        run_speed(ROUNDS)
        run_long(max(2, ROUNDS // 2), CTX)
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
