from . import ops
from .ops import *
from .autograd import Tensor, cpu, all_devices, no_grad

from . import init
from .init import ones, zeros, zeros_like, ones_like

from .backend_selection import *

# Lazy-load heavy subpackages to avoid circular imports at module init time.
# They are loaded on first attribute access (e.g. uniti.data, uniti.nn).
import importlib as _importlib
import sys as _sys

_LAZY_SUBMODULES = {"data", "nn", "optim"}
_LOADING = set()            # re-entrance guard


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        if name in _LOADING:
            # We are in a circular import; return the partially-initialized module
            # that importlib has already placed in sys.modules.
            fullname = f"{__name__}.{name}"
            mod = _sys.modules.get(fullname)
            if mod is not None:
                return mod
            raise ImportError(f"circular import while loading {fullname}")
        _LOADING.add(name)
        try:
            mod = _importlib.import_module(f".{name}", __name__)
        finally:
            _LOADING.discard(name)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
