"""Immutable identifiers needed to reproduce a benchmark result."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class VersionBundle:
    """Links a result to the code, configuration, and data used to produce it."""

    code_commit: str
    config_hash: str
    data_snapshot: str

    def as_dict(self) -> dict[str, str]:
        """Return a serialization-ready copy of this bundle."""
        return asdict(self)
