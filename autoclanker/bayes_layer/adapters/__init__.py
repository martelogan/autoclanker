from __future__ import annotations

from autoclanker.bayes_layer.adapters.autoresearch import AutoresearchAdapter
from autoclanker.bayes_layer.adapters.cevolve import CevolveAdapter
from autoclanker.bayes_layer.adapters.external import (
    JsonSubprocessAdapter,
    load_module_adapter,
)
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.adapters.protocols import (
    AdapterProbeResult,
    EvalLoopAdapter,
)
from autoclanker.bayes_layer.types import AdapterFailure, ValidAdapterConfig


def available_adapter_kinds() -> tuple[str, ...]:
    return ("fixture", "autoresearch", "cevolve", "python_module", "subprocess")


def load_adapter(config: ValidAdapterConfig) -> EvalLoopAdapter:
    if config.kind == "fixture":
        return FixtureAdapter(config)
    if config.kind == "python_module":
        if config.python_module is None:
            raise AdapterFailure("python_module adapters require python_module.")
        return load_module_adapter(
            config,
            kind=config.kind,
            module_name=config.python_module,
            execution_mode=config.mode,
            detail=f"Loaded python module adapter {config.python_module}",
        )
    if config.kind == "subprocess":
        return JsonSubprocessAdapter(config, kind=config.kind)
    if config.kind == "autoresearch":
        return AutoresearchAdapter(config)
    if config.kind == "cevolve":
        return CevolveAdapter(config)
    raise AdapterFailure(
        f"No builtin adapter implementation exists for kind {config.kind!r}."
    )


__all__ = [
    "AdapterProbeResult",
    "AutoresearchAdapter",
    "CevolveAdapter",
    "EvalLoopAdapter",
    "FixtureAdapter",
    "available_adapter_kinds",
    "load_adapter",
]
