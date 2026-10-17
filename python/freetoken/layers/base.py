from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def to(self, device: Any) -> "BaseOP":
        """Move every owned tensor to ``device`` (nn.Module.to analogue, for the CPU-hidden MTP
        path). Own tensors move even when underscore-held (a rope cos/sin cache lives there); nested
        ops recurse only under their public name so a shared tower attached as ``_embed`` stays put
        on its own device rather than being dragged along."""
        for name, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
            elif isinstance(value, BaseOP):
                if not name.startswith("_"):
                    value.to(device)
            elif isinstance(value, (list, tuple)):
                moved = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        moved.append(item.to(device))
                    elif isinstance(item, BaseOP):
                        item.to(device)
                        moved.append(item)
                    else:
                        moved.append(item)
                self.__dict__[name] = type(value)(moved) if isinstance(value, tuple) else moved
        return self

    def parameters(self):
        """Yield every owned tensor, recursively (nn.Module.parameters analogue), so callers can
        read ``next(op.parameters()).device`` for the op's current compute device."""
        for value in list(self.__dict__.values()):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, BaseOP):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        yield item
                    elif isinstance(item, BaseOP):
                        yield from item.parameters()

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
