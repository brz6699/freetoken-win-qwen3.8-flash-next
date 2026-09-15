"""Offline test of P4 (pixel content-hash prefix cache key).

1. media._content_id / _content_key: the content id is derived from the payload's sha1
   filename -- consistent for an identical image, different across images, within int32
   and far above the vocab. _content_key replaces ONLY the image_pad run and leaves the
   rest of the prompt bit-for-bit identical (independent tensor).
2. Radix content addressing (the invariant P4 rests on): insert with a content-keyed
   sequence, then match the SAME key -> full-length hit (KV reuse); match a DIFFERENT
   image's key -> stops at the shared text prefix (no false hit); raw text keys are
   unaffected.

  python -X utf8 verification/p4_content_key_test.py
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "package"))

FAIL = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def main() -> int:
    from freetoken.tokenizer import media

    def fake_path(seedhex: str) -> str:
        # mirrors _stage_payload's name: ft_mm_<sha1hex>.pt
        return os.path.join(tempfile.gettempdir(), "ft_mm_" + seedhex + ".pt")

    pathA = fake_path("ab" * 16)
    pathB = fake_path("cd" * 16)
    idA = media._content_id(pathA)
    idB = media._content_id(pathB)
    check("id.in_range", 1_000_000_000 <= idA < 2_000_000_000, f"idA={idA}")
    check("id.int32_safe", -2**31 <= idA < 2**31 and -2**31 <= idB < 2**31, "")
    check("id.consistent_same", media._content_id(pathA) == idA, "(same path -> same id)")
    check("id.differs_across", idA != idB, f"idA={idA} idB={idB}")

    PAD = 151655  # stand-in image_pad id
    ids = torch.tensor([10, 11, PAD, PAD, PAD, 12, 13], dtype=torch.int32)
    key = media._content_key(ids, PAD, pathA)
    check("key.shape_dtype", key.shape == ids.shape and key.dtype == ids.dtype,
          str(tuple(key.shape)))
    check("key.pad_replaced", bool((key[2:5] == idA).all()), f"got {key[2:5].tolist()}")
    check("key.nonpad_unchanged",
          bool((key[0:2] == ids[0:2]).all() and (key[5:7] == ids[5:7]).all()), "")
    check("key.is_copy", key.data_ptr() != ids.data_ptr(), "(independent tensor)")
    # the MODEL still sees the real pad ids (the key is a separate tensor)
    check("key.real_ids_intact", bool((ids == torch.tensor(
        [10, 11, PAD, PAD, PAD, 12, 13], dtype=torch.int32)).all()), "")

    # ------------------------------------------------------------------ #
    # 2. radix content addressing                                          #
    # ------------------------------------------------------------------ #
    from freetoken.kvcache.radix_cache import RadixPrefixCache

    dev = torch.device("cpu")
    text_pre = [10, 11]
    keyA = torch.tensor(text_pre + [idA, idA, idA, 12, 13], dtype=torch.int32)
    keyB = torch.tensor(text_pre + [idB, idB, idB, 12, 13], dtype=torch.int32)
    vals7 = torch.arange(7, dtype=torch.int32)  # page indices must match the key length

    # turn 1 caches image A; turn 2 (same image) must reuse the FULL prefix
    cache = RadixPrefixCache(dev, page_size=1)
    cache.insert_prefix(keyA, vals7)
    mA = cache.match_prefix(keyA)
    check("radix.same_hit", mA.cuda_handle.cached_len == 7,
          f"cached_len={mA.cuda_handle.cached_len} (want 7 = full reuse of image A's KV)")

    # a DIFFERENT image must not false-hit: only the shared text prefix matches
    cache2 = RadixPrefixCache(dev, page_size=1)
    cache2.insert_prefix(keyA, vals7)
    mB = cache2.match_prefix(keyB)
    check("radix.diff_no_false_hit", mB.cuda_handle.cached_len == 2,
          f"cached_len={mB.cuda_handle.cached_len} (want 2 = text prefix only, image span differs)")

    # raw text requests are unaffected by the content-key machinery
    cache3 = RadixPrefixCache(dev, page_size=1)
    t1 = torch.tensor([10, 11, 40, 41, 42], dtype=torch.int32)
    vals5 = torch.arange(5, dtype=torch.int32)
    cache3.insert_prefix(t1, vals5)
    mT = cache3.match_prefix(t1.clone())
    check("radix.text_unchanged", mT.cuda_handle.cached_len == 5, f"cached_len={mT.cuda_handle.cached_len}")
    mT2 = cache3.match_prefix(torch.tensor([10, 11, 99, 41, 42], dtype=torch.int32))
    check("radix.text_diverge", mT2.cuda_handle.cached_len == 2, f"cached_len={mT2.cuda_handle.cached_len}")

    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
