from __future__ import annotations

from autoclanker.bayes_layer.types import ValidAdapterConfig
from tests.fixtures.adapter_shim_common import (
    ContractShimAdapter,
    repo_shim_detail,
)

AUTOCLANKER_ADAPTER = None


def build_autoclanker_adapter(config: ValidAdapterConfig) -> ContractShimAdapter:
    return ContractShimAdapter(
        config,
        kind="autoresearch",
        execution_mode="local_repo_path",
        detail=repo_shim_detail("autoresearch", config.repo_path),
    )
