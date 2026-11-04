from __future__ import annotations

import os
import sys
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


ENV_PREFIX = "FREETOKEN_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)
EnvStr = partial(EnvVar[str], fn=str)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    # None = unset -> resolved from the model's generation_config.json sampling defaults
    # (sglang's sampling_defaults='model'); set the env var to override.
    SHELL_TOP_K = EnvInt(None)
    SHELL_TOP_P = EnvFloat(None)
    SHELL_TEMPERATURE = EnvFloat(None)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    # GatedDeltaNet recurrent (SSM) state dtype: float32 (default) | bfloat16 | float16.
    # fp32 matches the Qwen3.x configs (mamba_ssm_dtype); fp16/bf16 halves the GDN state
    # pool at some precision cost on the long recurrence (mirrors SGLang's mamba_ssm_dtype).
    MAMBA_SSM_DTYPE = EnvStr("float32")

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{ENV_PREFIX}{attr_name}")


_FT_DEFS_H = (
    "#ifndef FT_DEFS_H_\n"
    "#define FT_DEFS_H_\n"
    "typedef unsigned short ushort;\n"
    "typedef unsigned char uchar;\n"
    "typedef unsigned int uint;\n"
    "typedef unsigned long ulong;\n"
    "#define __attribute__(x)\n"
    "#define __restrict__ __restrict\n"
    "#ifdef __cplusplus\n"
    "namespace std {\n"
    "inline constexpr unsigned long long max(unsigned long long a, unsigned long b) { return a > b ? a : b; }\n"
    "inline constexpr unsigned long long max(unsigned long a, unsigned long long b) { return b > a ? b : a; }\n"
    "}\n"
    "#endif\n"
    "#endif\n"
)


def _ensure_jit_defs_header(base: str) -> None:
    """Materialize the forced-include shim header under the flashinfer JIT root.

    cudafe++ on this CUDA 13.1 box parses host stubs in MSVC-emulation mode and
    rejects flashinfer's ``FLASHINFER_INLINE`` expansion shape
    (``inline __attribute__((always_inline)) __device__``): an attribute sitting
    between ``inline`` and ``__device__`` kills its parser (~100 errors, build
    aborted). Expanding ``__attribute__(x)`` to nothing before the headers are
    parsed restores the accepted shape. The typedefs cover ``ushort``/``uchar``
    which MSVC CRT headers leave undefined in C++ mode while flashinfer's kernels
    use them freely. kernel/winninja.apply() adds the matching ``-include`` to the
    ninja cuda rule; the path must stay inside the short JIT root so it never hits
    this box's MAX_PATH limit. The same header is force-included on the cl-only
    host-stub pass via ``-Xcompiler /FI``, since nvcc's ``-include`` does not reach
    that second compile.
    """
    hdr = os.path.join(base, "ftdefs.h")
    try:
        os.makedirs(base, exist_ok=True)
        try:
            with open(hdr, "r", encoding="utf-8") as fh:
                if fh.read() == _FT_DEFS_H:
                    return
        except OSError:
            pass
        with open(hdr, "w", encoding="utf-8") as fh:
            fh.write(_FT_DEFS_H)
    except OSError:
        pass


def _bootstrap_windows_host_env() -> None:
    """Host-env prep so a plain `python -m freetoken` / ft.exe launch behaves like the
    exp launcher on Windows. flashinfer's JIT shells out to a bare ``ninja`` (which
    lives in the venv's Scripts dir next to the interpreter) and nvcc derives its root
    from CUDA_HOME whose default carries spaces that CreateProcess rejects unquoted.
    The peripheral flashinfer kernels additionally JIT with gcc-style link flags an
    MSVC-only host cannot satisfy, so default those ops to the triton kernels; the
    b12x fused-MoE gate is unaffected (see kernel/backend.periph_flashinfer_ok).
    setdefault throughout: a real env value always wins. Workers on spawn re-import
    this module and inherit the env."""
    if os.name != "nt":
        return
    scripts = os.path.dirname(os.path.abspath(sys.executable))
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if scripts not in path_parts:
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("FREETOKEN_FLASHINFER_PERIPH", "0")
    # flashinfer builds its JIT root as <FLASHINFER_WORKSPACE_BASE>/.cache/flashinfer/
    # <ver>/<arch>/{generated,cached_ops}/<module>/... ; with the compacted short module
    # names (kernel/winninja.compact_jit_uris) every resulting path stays ~100 chars, so
    # a plain short root fits this box's MAX_PATH (LongPathsEnabled=0) without the \\?\
    # prefix -- which cmd.exe treats as UNC ("UNC paths are not supported") and cl then
    # fails to open with C1083. A real env value still wins.
    drive = os.path.splitdrive(os.path.abspath(sys.executable))[0] or "C:"
    ws_base = os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", f"{drive}\\fi")
    _ensure_jit_defs_header(ws_base)
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and " " in cuda_home:
        import ctypes

        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(cuda_home, buf, 1024)
        if 0 < n < 1024:
            os.environ["CUDA_HOME"] = buf.value


_bootstrap_windows_host_env()

ENV = EnvClassSingleton()
