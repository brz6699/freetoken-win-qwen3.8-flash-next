"""Client-supplied image handling: ref collection, the accept gate, and byte fetching.

Raises plain ValueError on bad input; the server layer maps it to its wire error type.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from freetoken.server.args import ServerArgs

_MAX_IMAGE_BYTES = 32 << 20


def collect_image_refs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pop the image refs out of rendered messages in prompt order; the bare {"type": "image"} parts stay for the chat template."""
    refs: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                ref = part.pop("freetoken_ref", None)
                if ref:
                    refs.append(ref)
    return refs


def image_reject_reason(config: ServerArgs) -> str | None:
    """None when this server can accept image inputs, else the client-facing reason."""
    if "image" in config.served_modalities:
        return None
    if config.mm.text_model_only:
        return "image input is disabled on this server (--text-model-only)"
    if "vision" in config.mm.disabled_encoders:
        return "image input is disabled on this server (--mm-disable)"
    return "the served model has no vision support"


def _check_media_domain(url: str, config: ServerArgs) -> None:
    """Reject a URL whose hostname is not in --allowed-media-domains; empty allowlist admits any domain."""
    from urllib.parse import urlparse

    raw = config.allowed_media_domains
    allowed = {d.strip().lower().rstrip(".") for d in raw.split(",") if d.strip()}
    if not allowed:
        return
    host = (urlparse(url).hostname or "").rstrip(".")
    if host not in allowed:
        raise ValueError(
            f"the URL must be from one of the allowed domains: {sorted(allowed)}; "
            f"input URL domain: {host or '<none>'}"
        )


def _load_local_media(url: str, config: ServerArgs) -> bytes:
    """Read a file:// ref; the resolved path must be a strict subpath of --allowed-local-media-path."""
    from pathlib import Path
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    root = config.allowed_local_media_path
    if not root:
        raise ValueError("cannot load local files without --allowed-local-media-path")
    spec = urlparse(url)
    filepath = Path(url2pathname((spec.netloc or "") + (spec.path or "")))
    # resolve() follows symlinks, so a link inside the root escaping it is rejected too
    resolved = filepath.resolve()
    if Path(root).resolve() not in resolved.parents:
        raise ValueError(
            f"the file path {filepath} must be a subpath of "
            f"--allowed-local-media-path {root}"
        )
    return resolved.read_bytes()


async def fetch_image_bytes(refs: list[dict[str, Any]], config: ServerArgs) -> list[bytes]:
    """Resolve collected image refs (URLs / base64) to raw bytes, in order.

    HTTP refs are fetched concurrently over one shared client (order of results is
    preserved); every other kind resolves inline."""
    out: list[bytes] = [b""] * len(refs)
    http_slots: list[tuple[int, str]] = []
    for i, ref in enumerate(refs):
        data = ref.get("data") or ""
        try:
            if ref.get("kind") == "b64":
                out[i] = base64.b64decode(data)
            elif data.startswith("data:"):
                out[i] = base64.b64decode(data.split(",", 1)[1])
            elif data.startswith(("http://", "https://")):
                _check_media_domain(data, config)
                http_slots.append((i, data))
            elif data.startswith("file://"):
                out[i] = _load_local_media(data, config)
            else:
                raise ValueError(
                    "unsupported image source (expect http(s)/file url or base64)"
                )
        except Exception as exc:  # noqa: BLE001 -- input-driven, client-classifiable
            raise ValueError(f"could not load image: {exc}") from exc
    if http_slots:
        import httpx

        # redirect targets are not re-checked against the allowlist
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            results = await asyncio.gather(
                *(client.get(url) for _slot, url in http_slots), return_exceptions=True
            )
            for (i, _url), res in zip(http_slots, results):
                try:
                    if isinstance(res, Exception):
                        raise res
                    res.raise_for_status()
                    if len(res.content) > _MAX_IMAGE_BYTES:
                        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
                    out[i] = res.content
                except Exception as exc:  # noqa: BLE001 -- input-driven, client-classifiable
                    raise ValueError(f"could not load image: {exc}") from exc
    return out


__all__ = ["collect_image_refs", "fetch_image_bytes", "image_reject_reason"]
