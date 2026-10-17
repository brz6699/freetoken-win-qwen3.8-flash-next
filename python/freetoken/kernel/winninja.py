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


__all__ = ["apply", "trim_route_pack_smem", "allow_ultrawide_fc2_tile"]
