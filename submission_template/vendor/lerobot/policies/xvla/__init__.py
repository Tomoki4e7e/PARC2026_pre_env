from .configuration_xvla import XVLAConfig


_LAZY_IMPORTS = {
    ")": "processor_xvla",
    "XVLAAddDomainIdProcessorStep": "processor_xvla",
    "XVLAImageNetNormalizeProcessorStep": "processor_xvla",
    "XVLAImageToFloatProcessorStep": "processor_xvla",
    "XVLAPolicy": "modeling_xvla",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "XVLAConfig",
    "XVLAPolicy",
    "XVLAAddDomainIdProcessorStep",
    "XVLAImageNetNormalizeProcessorStep",
    "XVLAImageToFloatProcessorStep",
]
