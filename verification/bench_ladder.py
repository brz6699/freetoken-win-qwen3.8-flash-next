"""Staged concurrency throughput ladder against a live serve (default :8001).

Two design points driven by measured artifacts of naive harnesses:

1. Decode rate is isolated with a DIFFERENTIAL method (out=SHORT vs out=LONG on
   the same prompt; rate = dTokens/dTime). Short outputs let TTFT dominate the
   average; the diff cancels it.

2. With max_running_req=4 on the server, a bs4 client often sees running-req=1
   per step (requests stagger and finish before the next arrives). So all
   requests in a batch are released by a threading.Barrier for simultaneous
   arrival, AND we test bs > max_running_req (8, 16): the excess queues and
   feeds the 4 running slots back-to-back, which is the real steady-state
   pressure a busy agent farm produces.

  python bench_ladder.py [base_url] [long_out] [sizes_csv]
"""
import json
import sys
import time
import threading
import urllib.request
import concurrent.futures as cf

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
LONG = int(sys.argv[2]) if len(sys.argv) > 2 else 512
SHORT = 32
SIZES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else
         ["10000", "50000", "100000", "200000", "500000", "1000000"])]
BSS = (1, 4, 8, 16)

# observed tokenizer ratio on this model: chars/token ~ 4.5 (repetitive unit text)
CPU_PER_TOK = 4.5


def build(ntok: int) -> str:
    unit = ("The cache arena keeps KV left-anchored and MoE slots right-anchored so the "
            "boundary slides without reallocating; the next agent turn reuses parked rows "
            "instead of re-prefilling the whole prefix. ")
    chars = int(ntok * CPU_PER_TOK)
    reps = max(1, chars // len(unit))
    return unit * reps


def one(prompt: str, max_tokens: int):
    payload = {"model": "qwen",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    body = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
    ms = (time.monotonic() - t0) * 1000
    u = body.get("usage", {})
    ctok = u.get("completion_tokens") or max_tokens
    return ms, ctok, u.get("prompt_tokens", 0)


def conc(n, prompt, max_tokens):
    """All n requests block on a barrier, then fire together."""
    barrier = threading.Barrier(n)

    def fired():
        barrier.wait(timeout=60)
        return one(prompt, max_tokens)

    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.monotonic()
        futs = [ex.submit(fired) for _ in range(n)]
        r = [f.result() for f in cf.as_completed(futs)]
    wall = max((time.monotonic() - t0) * 1000, 1e-6)
    return wall, r


def bs_rate(bs, prompt):
    """Differential aggregate decode rate at concurrency bs + latency spread."""
    wall_s, rs = conc(bs, prompt, SHORT)
    wall_l, rl = conc(bs, prompt, LONG)
    dt = wall_l - wall_s
    dn = sum(x[1] for x in rl) - sum(x[1] for x in rs)
    rate = dn / (dt / 1000.0) if dt > 0 and dn > 0 else 0.0
    lat = sorted(x[0] for x in rl)
    return rate, lat


def stage(ntok: int) -> None:
    p = build(ntok)
    _, _, ptok = one(p, SHORT)          # warm + read true prompt size
    print(f"\n--- input~{ntok} (prompt={ptok}t) out_long={LONG}t ---")

    rates = {}
    for bs in BSS:
        rate, lat = bs_rate(bs, p)
        rates[bs] = rate
        spread = lat[-1] - lat[0] if lat else 0
        print(f"  bs={bs:>2} agg={rate:.1f}tok/s x{rate/max(rates[1],1e-6):.2f} "
              f"lat_min/max={lat[0]:.0f}/{lat[-1]:.0f}ms spread={spread:.0f}ms")

    c_ms, _, _ = one(p, SHORT)
    rr = [one(p, SHORT)[0] for _ in range(2)]
    rep = sum(rr) / len(rr)
    print(f"  ttft cold={c_ms:.0f}ms repeat={rep:.0f}ms speedup_x={c_ms/max(rep,1):.2f}")


def main() -> int:
    for n in SIZES:
        stage(n)
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
