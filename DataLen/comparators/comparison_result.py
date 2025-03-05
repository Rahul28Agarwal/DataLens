from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComparisonResult:
    """Stores results of a comparison between two DataFrame columns."""

    first_data_col: str | None = None
    second_data_col: str | None = None
    comparison_type: str | None = None
    is_matched: bool | None = None
    first_data_count: int | None = None
    second_data_count: int | None = None
    missing: dict[str, list[str | int]] = field(default_factory=dict)
    meta: str | None = None
    additional_info: dict[str, int | str] = field(default_factory=dict)

    def add_key(self, key: str, default_value: int | str | list[str] | None = None) -> None:  # noqa: D102
        setattr(self, key, default_value)

    def to_dict(self) -> dict:
        """Convert the result to a dictionary."""
        return self.__dict__
