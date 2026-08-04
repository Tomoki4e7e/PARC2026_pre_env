#!/usr/bin/env python

from .configuration_eo1 import EO1Config


_LAZY_IMPORTS = {
    "EO1Policy": "modeling_eo1",
    "make_eo1_pre_post_processors": "processor_eo1",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = ["EO1Config", "EO1Policy", "make_eo1_pre_post_processors"]
