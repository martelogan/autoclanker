from __future__ import annotations

import json

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

_T = TypeVar("_T", bound=Callable[..., object])


class _CoverableCallable(Protocol):
    __autoclanker_requirement_ids__: tuple[str, ...]

    def __call__(self, *args: object, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class Requirement:
    requirement_id: str
    gate: str
    description: str
    status: str


def _matrix_path() -> Path:
    return Path(__file__).with_name("compliance_matrix.json")


def load_requirement_matrix() -> tuple[Requirement, ...]:
    payload = cast(object, json.loads(_matrix_path().read_text(encoding="utf-8")))
    if not isinstance(payload, list):
        raise AssertionError("compliance_matrix.json must contain a list.")
    requirements: list[Requirement] = []
    for item in cast(list[object], payload):
        if not isinstance(item, dict):
            raise AssertionError("Each compliance matrix entry must be an object.")
        mapping = cast(dict[str, object], item)
        requirements.append(
            Requirement(
                requirement_id=str(mapping["requirement_id"]),
                gate=str(mapping["gate"]),
                description=str(mapping["description"]),
                status=str(mapping["status"]),
            )
        )
    return tuple(requirements)


def covers(*requirement_ids: str) -> Callable[[_T], _T]:
    normalized = tuple(requirement_ids)

    def decorator(func: _T) -> _T:
        existing = cast(
            tuple[str, ...],
            getattr(func, "__autoclanker_requirement_ids__", ()),
        )
        cast(_CoverableCallable, func).__autoclanker_requirement_ids__ = (
            existing + normalized
        )
        return func

    return decorator


__all__ = ["Requirement", "covers", "load_requirement_matrix"]
