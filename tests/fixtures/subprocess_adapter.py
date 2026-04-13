from __future__ import annotations

import json
import sys

from typing import cast

from autoclanker.bayes_layer.registry import build_fixture_registry
from autoclanker.bayes_layer.types import (
    AdapterKind,
    AdapterMode,
    EvalContractSnapshot,
    EvalExecutionContext,
    GeneStateRef,
    IsolationMode,
    JsonValue,
    ValidAdapterConfig,
    to_json_value,
)
from tests.fixtures.adapter_shim_common import ContractShimAdapter


def _require_object(value: object, *, message: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise SystemExit(message)
    return cast(dict[str, JsonValue], value)


def _require_string(value: object, *, message: str) -> str:
    if not isinstance(value, str):
        raise SystemExit(message)
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise SystemExit("integer field contained an unsupported value")


def _gene_state_refs(items: list[JsonValue]) -> tuple[GeneStateRef, ...]:
    refs: list[GeneStateRef] = []
    for item in items:
        mapping = _require_object(item, message="genotype items must be objects")
        refs.append(
            GeneStateRef(
                gene_id=_require_string(
                    mapping.get("gene_id"),
                    message="genotype items must include gene_id",
                ),
                state_id=_require_string(
                    mapping.get("state_id"),
                    message="genotype items must include state_id",
                ),
            )
        )
    return tuple(refs)


def _eval_contract_snapshot(value: object) -> EvalContractSnapshot:
    mapping = _require_object(
        value, message="execution_context.contract must be an object"
    )
    return EvalContractSnapshot(
        contract_digest=_require_string(
            mapping.get("contract_digest"),
            message="execution_context.contract.contract_digest is required",
        ),
        benchmark_tree_digest=_require_string(
            mapping.get("benchmark_tree_digest"),
            message="execution_context.contract.benchmark_tree_digest is required",
        ),
        eval_harness_digest=_require_string(
            mapping.get("eval_harness_digest"),
            message="execution_context.contract.eval_harness_digest is required",
        ),
        adapter_config_digest=_require_string(
            mapping.get("adapter_config_digest"),
            message="execution_context.contract.adapter_config_digest is required",
        ),
        environment_digest=_require_string(
            mapping.get("environment_digest"),
            message="execution_context.contract.environment_digest is required",
        ),
        workspace_snapshot_id=_optional_string(mapping.get("workspace_snapshot_id")),
        workspace_snapshot_mode=_optional_string(
            mapping.get("workspace_snapshot_mode")
        ),
        captured_paths=cast(dict[str, JsonValue] | None, mapping.get("captured_paths")),
        captured_at=_optional_string(mapping.get("captured_at")),
    )


def _execution_context(value: object) -> EvalExecutionContext | None:
    if value is None:
        return None
    mapping = _require_object(
        value,
        message="execution_context payload must be an object",
    )
    isolation_mode = _require_string(
        mapping.get("isolation_mode"),
        message="execution_context.isolation_mode is required",
    )
    if isolation_mode not in {"git_worktree", "copy", "fixture"}:
        raise SystemExit(
            "execution_context.isolation_mode must be git_worktree, copy, or fixture"
        )
    return EvalExecutionContext(
        session_id=_require_string(
            mapping.get("session_id"),
            message="execution_context.session_id is required",
        ),
        era_id=_require_string(
            mapping.get("era_id"),
            message="execution_context.era_id is required",
        ),
        contract=_eval_contract_snapshot(mapping.get("contract")),
        isolation_mode=cast(IsolationMode, isolation_mode),
        workspace_root=_optional_string(mapping.get("workspace_root")),
        seed=_integer(mapping.get("seed")),
        replication_index=_integer(mapping.get("replication_index")),
    )


def main() -> int:
    payload = _require_object(
        json.loads(sys.stdin.read()),
        message="expected JSON request object",
    )
    operation = _require_string(
        payload.get("operation"),
        message="request must include operation",
    )
    config_payload = _require_object(
        payload.get("config"),
        message="config payload must be an object",
    )
    command_raw = config_payload.get("command", [])
    if not isinstance(command_raw, list):
        raise SystemExit("command payload must be a list")
    config = ValidAdapterConfig(
        kind=cast(
            AdapterKind,
            _require_string(
                config_payload.get("kind"), message="config.kind is required"
            ),
        ),
        mode=cast(
            AdapterMode,
            _require_string(
                config_payload.get("mode"), message="config.mode is required"
            ),
        ),
        session_root=_require_string(
            config_payload.get("session_root"),
            message="config.session_root is required",
        ),
        allow_missing=bool(config_payload.get("allow_missing", False)),
        repo_path=_optional_string(config_payload.get("repo_path")),
        python_module=_optional_string(config_payload.get("python_module")),
        command=tuple(str(item) for item in command_raw),
        config_path=_optional_string(config_payload.get("config_path")),
        base_dir=_optional_string(config_payload.get("base_dir")),
    )
    adapter = ContractShimAdapter(
        config=config,
        kind=str(config.kind),
        execution_mode="subprocess_cli",
        detail=f"Loaded subprocess shim {config.command[0]}",
    )
    if operation == "probe":
        response = adapter.probe()
    elif operation == "capture_eval_contract":
        response = adapter.capture_eval_contract()
    elif operation == "build_registry":
        response = build_fixture_registry().to_dict()
    elif operation == "materialize_candidate":
        genotype = payload.get("genotype")
        if not isinstance(genotype, list):
            raise SystemExit("materialize_candidate requires genotype list")
        response = adapter.materialize_candidate(
            _gene_state_refs(genotype),
        )
    elif operation == "evaluate_candidate":
        genotype = payload.get("genotype")
        if not isinstance(genotype, list):
            raise SystemExit("evaluate_candidate requires genotype list")
        response = adapter.evaluate_candidate(
            era_id=_require_string(payload.get("era_id"), message="era_id is required"),
            candidate_id=_require_string(
                payload.get("candidate_id"),
                message="candidate_id is required",
            ),
            genotype=_gene_state_refs(genotype),
            seed=_integer(payload.get("seed")),
            replication_index=_integer(payload.get("replication_index")),
            execution_context=_execution_context(payload.get("execution_context")),
        )
    elif operation == "commit_candidate":
        response = adapter.commit_candidate(
            _require_string(
                payload.get("candidate_id"),
                message="candidate_id is required",
            )
        )
    else:
        raise SystemExit(f"Unsupported operation {operation!r}")
    print(json.dumps(to_json_value(response), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
