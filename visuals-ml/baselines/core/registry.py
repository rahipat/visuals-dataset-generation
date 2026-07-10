"""
Model registry. Each baseline registers itself with @register_model("name"),
and the runner builds one by name from a config's `model:` key.
"""

from typing import Dict, Type

_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"Model '{name}' is already registered to {_REGISTRY[name]}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def build_model(cfg: dict):
    name = cfg.get("model")
    if name is None:
        raise KeyError("Config is missing required key 'model'")
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {available_models()}\n"
            "(Did you forget to import it in baselines/models/__init__.py?)"
        )
    return _REGISTRY[name].from_config(cfg)


def available_models():
    return sorted(_REGISTRY)
