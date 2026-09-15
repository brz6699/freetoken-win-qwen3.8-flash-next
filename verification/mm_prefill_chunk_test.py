"""Offline test of the mm single-chunk prefill admission (scheduler/prefill.py).

After the guard-max crash (32768-patch image -> 8214-token prompt > 8192
max-extend-tokens -> NotImplementedError killed the scheduler process), the
adder must: (a) still hard-fail if a prompt over the FULL budget slips past
admission, (b) roll its budget charges back and return None (retry on a fresh
pass) when only THIS pass's leftover is short, (c) keep chunking plain text
prompts, (d) admit whole mm prompts that fit the fresh budget.

  python -X utf8 verification/mm_prefill_chunk_test.py
"""

import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "package"))
from freetoken.core import SamplingParams  # noqa: E402
from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder  # noqa: E402
from freetoken.scheduler.utils import PendingReq  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


class FakeCM:
    swa_paged = False
    prefill_chunk_align = 1
    page_size = 64
    sliding_window_size = 10_000
    swa_available_size = 0

    def __init__(self):
        self.swa_paged = False
        self.prefill_chunk_align = 1


class FakeTM:
    def __init__(self):
        self._pool = torch.zeros(4, 16384, dtype=torch.int32)

    @property
    def token_pool(self):
        return self._pool


class FakeCH:
    cached_len = 0


def make_adder(token_budget, full_budget=None):
    return PrefillAdder(
        token_budget=token_budget,
        reserved_size=0,
        cache_manager=FakeCM(),
        table_manager=FakeTM(),
        full_budget=full_budget if full_budget is not None else token_budget,
    )


def make_req(n, mm):
    sp = SamplingParams()
    sp.max_tokens = 16
    return PendingReq(
        uid=1,
        input_ids=torch.zeros(n, dtype=torch.int32),
        sampling_params=sp,
        mm_embeds=torch.zeros(8, 2560) if mm else None,
    )


# (a) mm prompt over the FULL budget: the adder's loud guard still fires
#     (in production the scheduler rejects it at admission with a 400 first).
adder = make_adder(8192)
try:
    adder._add_one_req(make_req(8214, mm=True), FakeCH(), 0, 0)
    check("a.over_full_guard", False, "(no exception)")
except NotImplementedError as e:
    check("a.over_full_guard", "max-extend-tokens" in str(e), str(e)[:80])

# (b) mm prompt that fits a fresh pass but not this pass's leftover:
#     None + full rollback of token_budget / reserved_size / reserved_swa.
adder = make_adder(8192)
adder.token_budget = 3000  # a same-pass predecessor consumed 5192
r = adder._add_one_req(make_req(5000, mm=True), FakeCH(), 0, 0)
check("b.midpass_retry", r is None, f"got {type(r).__name__ if r is not None else None}")
check("b.rollback_budget", adder.token_budget == 3000, f"got {adder.token_budget}")
check("b.rollback_reserved", adder.reserved_size == 0, f"got {adder.reserved_size}")
check("b.rollback_swa", adder.reserved_swa == 0, f"got {adder.reserved_swa}")

# (c) plain text prompt over this pass's leftover still CHUNKS (regression).
adder = make_adder(8192)
adder.token_budget = 3000
r = adder._add_one_req(make_req(5000, mm=False), FakeCH(), 0, 0)
check("c.text_chunks", isinstance(r, ChunkedReq), f"got {type(r).__name__}")
check("c.text_charged", adder.token_budget == 0, f"got {adder.token_budget}")

# (d) mm prompt that fits the fresh budget is admitted whole (a plain Req).
from freetoken.core import Req  # noqa: E402

adder = make_adder(8192)
r = adder._add_one_req(make_req(4000, mm=True), FakeCH(), 0, 0)
check(
    "d.mm_whole_admit",
    r is not None and type(r) is Req,
    f"got {type(r).__name__ if r is not None else None}",
)

# (e) swa cap collapses the chunk to 0 for an mm prompt: None + rollback, no crash.
class SwaCM(FakeCM):
    def __init__(self):
        super().__init__()
        self.swa_paged = True
        self.swa_available_size = 0

adder = PrefillAdder(
    token_budget=8192,
    reserved_size=0,
    cache_manager=SwaCM(),
    table_manager=FakeTM(),
    full_budget=8192,
)
r = adder._add_one_req(make_req(5000, mm=True), FakeCH(), 0, 0)
check("e.swa_zero_retry", r is None, f"got {type(r).__name__ if r is not None else None}")
check("e.swa_rollback", adder.token_budget == 8192 and adder.reserved_swa == 0,
      f"budget={adder.token_budget} swa={adder.reserved_swa}")

print()
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("RESULT: ALL PASS")
