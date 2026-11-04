"""Windows-only fix for flashinfer's JIT build.ninja, applied LAZILY after CUDA init.

flashinfer 0.7.1's ``JitSpecNvcc`` emits a build.ninja with three Windows defects that each
abort the build:
  1. drive-letter colons in build/default paths (C:\\...) -- ninja's path reader stops at ':',
     so the drive letter is mis-parsed as the whole target ("expected build command name");
  2. ``$nvcc_launcher`` / ``$cxx_launcher`` default to the empty string, so every build command
     expands with a LEADING SPACE, which Windows CreateProcess rejects;
  3. a CUDA root path with spaces, unquoted in the command line.

The patch rewrites the just-written build.ninja: it strips the empty launcher refs and escapes
the drive colons on build/default lines as ``$:`` (ninja's literal-colon escape). The spaced
CUDA root is handled upstream by pointing ``CUDA_HOME`` at the 8.3 short name.

CRITICAL: ``apply()`` must run only AFTER this process has initialized a CUDA context.
``import flashinfer.jit.core`` (done inside apply) initializes CUDA as a side effect, and the
engine asserts it was the first thing to touch CUDA (``engine.py``:
``assert not torch.cuda.is_initialized()``). So the engine calls apply() right after
``bind_assigned_gpu``, NOT at launcher/import time. Idempotent; a no-op on non-Windows or if
flashinfer is unavailable. The b12x pack's fp4 block-scale swizzle builds a flashinfer .cu
module via this build.ninja path, so the patch must be in place before the offload MoE cache
is constructed.
"""
from __future__ import annotations

import os

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED or os.name != "nt":
        return
    try:
        import re
        from flashinfer.jit.core import JitSpecNvcc
    except Exception:
        return
    if getattr(JitSpecNvcc.write_ninja, "_ft_win_patched", False):
        _APPLIED = True
        return

    drive_colon = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]):([\\/])")
    launcher = re.compile(r"^(cxx_launcher|nvcc_launcher)\s*=\s*(.*)$")
    orig_write_ninja = JitSpecNvcc.write_ninja

    def write_ninja_win(self):
        orig_write_ninja(self)
        p = self.ninja_path
        text = p.read_text()
        empty = {m.group(1) for m in map(launcher.match, text.splitlines())
                 if m and m.group(2).strip() == ""}
        for name in empty:
            text = text.replace("$" + name + " ", "").replace("$" + name, "")
        out = []
        for line in text.split("\n"):
            if line.lstrip().startswith(("build ", "default ")):
                line = drive_colon.sub(r"\1$:\2", line)
            # Four host-side fixes on the cuda compile rule
            # (`command = $nvcc ...`; the cuda_home/nvcc var lines carry no space):
            #  - `-include ftdefs.h`: cudafe++'s MSVC-emulation parser rejects
            #    flashinfer's FLASHINFER_INLINE expansion shape and MSVC CRT
            #    lacks the ushort typedef; the shim header fixes both (written by
            #    env._ensure_jit_defs_header under the same short JIT root).
            #  - `-Xcompiler /FI<defs>`: nvcc passes -include only to the .cu
            #    preprocess pass; the cl compile of cudafe's host stub re-includes
            #    the headers without it, so the shim must also arrive via /FI.
            #  - cccl headers #error (C1189) under the traditional preprocessor;
            #    cccl's own escape hatch macro silences it without changing
            #    preprocessing semantics (/Zc:preprocessor itself breaks nvcc's
            #    generated host stubs, so it stays off). Plain -D reaches both
            #    host and device passes.
            #  - /utf-8 stops C4819 spam under this box's cp936.
            if line.lstrip().startswith("command = ") and "$nvcc " in line \
                    and "CCCL_IGNORE_MSVC_TRADITIONAL" not in line:
                base = os.environ.get("FLASHINFER_WORKSPACE_BASE") or "C:\\fi"
                defs = os.path.join(base, "ftdefs.h")
                line = line.replace(
                    "$nvcc ",
                    f'$nvcc -include "{defs}" '
                    "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING "
                    "-Xcompiler /utf-8 "
                    f"-Xcompiler /FI{defs} ",
                    1,
                )
            out.append(line)
        p.write_text("\n".join(out))

    write_ninja_win._ft_win_patched = True
    JitSpecNvcc.write_ninja = write_ninja_win
    _APPLIED = True


def allow_ultrawide_fc2_tile() -> None:
    """Accept the tc-decode ultra-wide FC2 tile (tile_k=32, tile_n=512) at re-validation.

    flashinfer's compile-time tile selection deliberately picks fc2 tile_k=32 /
    tile_n=512 for small-M tc-decode (validating the smem footprint directly, since 32 is
    below the generic tile_k>=64 fits-floor) to equalize FC1/FC2 wave counts. But the
    custom-op launch boundary re-pins that geometry via force_tile_config, and
    ``_candidate_tile_fits`` hard-rejects any tile_k < 64 -- so the bs>=2 CUDA-graph
    capture path dies with "force_tile_config fc2 tile (tile_k=32, tile_n=512) does not
    fit problem N/K=2560/640". The override is only ever reached for hidden % 512 == 0
    and intermediate % 32 == 0, so accept that exact geometry here when it actually fits
    (smem budget + register-table entry exist); everything else falls through to the
    original check unchanged.

    Same lazy timing rule as apply(): import flashinfer only after the CUDA context is up.
    Idempotent.
    """
    if os.name != "nt":
        return
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_kernel as kw
    except Exception:
        return
    if getattr(kw._candidate_tile_fits, "_ft_ultrawide_fc2_patched", False):
        return
    orig = kw._candidate_tile_fits

    def patched(
        *,
        problem_n,
        problem_k,
        cta_m_blocks,
        tile_n,
        tile_k,
        cta_threads,
        max_shared_mem,
        scale_format="e4m3_k16",
        weight_layout="packed",
        allow_logical_tail=False,
    ):
        if (
            orig(problem_n=problem_n, problem_k=problem_k, cta_m_blocks=cta_m_blocks,
                 tile_n=tile_n, tile_k=tile_k, cta_threads=cta_threads,
                 max_shared_mem=max_shared_mem, scale_format=scale_format,
                 weight_layout=weight_layout, allow_logical_tail=allow_logical_tail)
        ):
            return True
        try:
            if (
                int(tile_n) != 512
                or int(tile_k) != 32
                or int(cta_threads) != 256
                or int(problem_n) % 512 != 0
                or int(problem_k) % 32 != 0
                or kw._shared_memory_footprint(
                    cta_m_blocks=cta_m_blocks, tile_n=512, tile_k=32,
                    scale_format=scale_format, weight_layout=weight_layout,
                ) > int(max_shared_mem)
            ):
                return False
            kw._w4a16_num_regs(
                cta_threads=256,
                cta_m_blocks=cta_m_blocks,
                cta_n_blocks=512 // 16,
                cta_k_blocks=32 // 16,
                uses_m_block_8=cta_m_blocks == 1,
                weight_layout=weight_layout,
            )
            return True
        except Exception:
            return False

    patched._ft_ultrawide_fc2_patched = True
    kw._candidate_tile_fits = patched


def force_ninja_output_encoding() -> None:
    """Decode flashinfer's captured ninja output as UTF-8 with lossy fallback.

    ``cpp_ext.run_ninja`` captures ninja's output with ``text=True`` and no explicit
    encoding, so Python decodes with the locale codec (gbk/cp936 on this zh-CN box).
    The piped stream mixes encodings (ninja status lines + cl's localized cp936
    messages), and strict cp936 dies mid-capture with UnicodeDecodeError
    ("0xad ... illegal multibyte sequence") even though the ninja build itself
    succeeds. Replacing the function with an identical copy that pins
    ``encoding="utf-8", errors="replace"`` keeps the build outcome identical and makes
    the capture total. jit/core.py binds ``run_ninja`` by value at import, so the
    wrapper is installed in both namespaces. Same lazy timing rule as apply().
    """
    if os.name != "nt":
        return
    try:
        import subprocess as sp

        from flashinfer.jit import cpp_ext, core
    except Exception:
        return
    orig = cpp_ext.run_ninja
    if getattr(orig, "_ft_ninja_enc_patched", False):
        return

    def run_ninja_utf8(workdir, ninja_file, verbose, max_jobs=None):
        workdir.mkdir(parents=True, exist_ok=True)
        command = [
            "ninja", "-v",
            "-C", str(workdir.resolve()),
            "-f", str(ninja_file.resolve()),
        ]
        num_workers = max_jobs if max_jobs is not None else cpp_ext._get_num_workers()
        if num_workers is not None:
            command += ["-j", str(num_workers)]
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        try:
            sp.run(
                command,
                stdout=None if verbose else sp.PIPE,
                stderr=sp.STDOUT,
                cwd=str(workdir.resolve()),
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except sp.CalledProcessError as e:
            msg = "Ninja build failed."
            if e.output:
                msg += " Ninja output:\n" + e.output
            raise RuntimeError(msg) from e

    run_ninja_utf8._ft_ninja_enc_patched = True
    cpp_ext.run_ninja = run_ninja_utf8
    if hasattr(core, "run_ninja"):
        core.run_ninja = run_ninja_utf8


def compact_jit_uris() -> None:
    """Deterministically shorten flashinfer JIT module uris so ninja accepts the build paths.

    flashinfer names each attention JIT module with its full signature (``batch_prefill_
    with_kv_cache_dtype_q_bf16_..._kv_stride_equal`` ~188 chars). That uri becomes BOTH the
    ``generated/<uri>/`` source directory and the JitSpec name, and the spec name is doubled
    in the artifact path (``cached_ops/<name>/<name>.so``, jit/core.py:460-468), so even from
    a one-letter drive root the ``.so`` path is ~370-415 chars. Python-side writes survive
    that via the ``\\?\\``-prefixed FLASHINFER_WORKSPACE_BASE, but ninja applies its own
    ~260-char check after normalizing the prefix away and dies at the first Stat. Shortening
    the uri fixes both the artifact path and the source-directory path at once.

    The uri influences ONLY disk naming and the spec name -- generated .cu content comes
    purely from kwargs/jinja templates -- so replacing long uris with
    ``<family>_<md5[:10]>`` is safe: deterministic across runs (restarts still hit the JIT
    cache), unique per distinct signature, and the result (~24 chars) fits ninja's limit and
    even this box's plain MAX_PATH without any prefix.

    Callers import these symbols by value (decode.py / prefill.py / pod.py re-bind them at
    import time), so the wrapper is installed into every namespace currently holding the
    original. Same lazy timing rule as apply(): import flashinfer only after the CUDA
    context is up. Idempotent; no-op on non-Windows.
    """
    if os.name != "nt":
        return
    try:
        import hashlib
        import importlib
        import inspect

        from flashinfer.jit.attention import modules as M
    except Exception:
        return

    def compact(uri):
        if not isinstance(uri, str) or len(uri) <= 100:
            return uri
        family = "_".join(uri.split("_", 2)[:2])
        return family + "_" + hashlib.md5(uri.encode("utf-8")).hexdigest()[:10]

    holders = [M]
    for dotted in (
        "flashinfer.jit.attention",
        "flashinfer.jit",
        "flashinfer.decode",
        "flashinfer.prefill",
        "flashinfer.pod",
    ):
        try:
            holders.append(importlib.import_module(dotted))
        except Exception:
            pass

    for fname in (
        "gen_customize_batch_prefill_module",
        "gen_customize_batch_decode_module",
    ):
        fn = getattr(M, fname, None)
        if fn is None or getattr(fn, "_ft_uri_compact_patched", False):
            continue
        try:
            idx = list(inspect.signature(fn).parameters).index("uri")
        except Exception:
            continue

        def patched(*args, _orig=fn, _idx=idx, **kwargs):
            if "uri" in kwargs:
                kwargs = {**kwargs, "uri": compact(kwargs["uri"])}
            elif len(args) > _idx and isinstance(args[_idx], str):
                args = args[:_idx] + (compact(args[_idx]),) + args[_idx + 1:]
            return _orig(*args, **kwargs)

        patched._ft_uri_compact_patched = True
        for mod in holders:
            if getattr(mod, fname, None) is fn or getattr(mod, fname, None) is None:
                try:
                    setattr(mod, fname, patched)
                except Exception:
                    pass


def trim_route_pack_smem() -> None:
    """Shrink flashinfer b12x route-pack tile heights so CUDA-graph capture fits SM120 smem.

    The captured-path single-CTA prefix kernel sizes its shared-memory tile as
    BLOCK_E x BLOCK_T (BLOCK_E = next_pow2(num_experts) = 512 for this model). With
    flashinfer's defaults BLOCK_T = 256 -> 512 x 256 = 131072 B, over the RTX 5090
    (SM120) per-block cap of 101376 B -> triton OutOfResources at graph capture, which
    forced ``--cuda-graph-max-bs 0`` and its ~2x bs=1 decode launch-overhead tax.
    BLOCK_T = 128 -> 65536 B fits comfortably; the kernels tile in a loop over BLOCK_T
    so a smaller tile is correctness-neutral (only more iterations). Constants are read
    from module globals at each launch call, so a setattr here takes effect immediately.

    Same lazy timing rule as apply(): import flashinfer only after the CUDA context is
    up. No-op if flashinfer is unavailable or the constants already fit.
    """
    if os.name != "nt":
        return
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_route_pack as rp
    except Exception:
        return
    for name in ("_COUNT_BLOCK_T", "_SORT_BLOCK_T", "_POST_PREFIX_BLOCK_T"):
        try:
            cur = int(getattr(rp, name))
        except Exception:
            continue
        if cur > 128:
            try:
                setattr(rp, name, min(cur, 128))
            except Exception:
                pass


__all__ = [
    "apply",
    "trim_route_pack_smem",
    "allow_ultrawide_fc2_tile",
    "compact_jit_uris",
    "force_ninja_output_encoding",
]
