from __future__ import annotations

from autoclanker.bayes_layer.types import ValidAdapterConfig
from tests.fixtures.adapter_shim_common import ContractShimAdapter


def build_autoclanker_adapter(config: ValidAdapterConfig) -> ContractShimAdapter:
    return ContractShimAdapter(
        config,
        kind=config.kind,
        execution_mode="python_module",
        detail=f"Loaded python_module shim {config.python_module}",
    )
