"""Variable-registry loader.

The NFP variable registry (CSV) is the single source of truth for the whole
suite: which latent traits exist, which blocks they belong to, their scales,
and reversal flags. The GUI input form, the scoring pipeline, and the report
sections are all *generated* from this file rather than hand-coded, so the
software never drifts away from the psychometric groundwork.

The bundled ``data/registry/registry_sample.csv`` is a small placeholder.
Replace it with an export of the real 271-row master registry (same columns)
and everything downstream picks it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "var_code",   # unique prefixed code, e.g. PSY_RES
    "block",      # block prefix, e.g. PSY (one of the 13 registry blocks)
    "label",      # human-readable name
    "kind",       # latent_trait | item | outcome | derived
    "scale_min",  # numeric scale bounds (z-scores use -3 / 3 by convention)
    "scale_max",
    "reverse",    # 1 if reverse-keyed, else 0
    "description",
]


@dataclass(frozen=True)
class Registry:
    """Parsed registry plus convenience accessors."""

    table: pd.DataFrame

    @property
    def traits(self) -> pd.DataFrame:
        """Rows describing latent traits (what the simulator consumes)."""
        return self.table[self.table["kind"] == "latent_trait"]

    @property
    def trait_codes(self) -> list[str]:
        return self.traits["var_code"].tolist()

    @property
    def blocks(self) -> list[str]:
        return sorted(self.table["block"].unique().tolist())

    def label_for(self, var_code: str) -> str:
        row = self.table.loc[self.table["var_code"] == var_code]
        if row.empty:
            raise KeyError(f"Unknown var_code: {var_code}")
        return str(row.iloc[0]["label"])


def load_registry(path: str | Path) -> Registry:
    """Load and validate a registry CSV.

    Raises ``ValueError`` with a readable message if the file is malformed,
    so bad registries fail fast instead of corrupting a simulation run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    table = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Registry {path} is missing required columns: {missing}")

    dupes = table["var_code"][table["var_code"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"Registry {path} has duplicate var_codes: {dupes}")

    if not table["kind"].isin(["latent_trait", "item", "outcome", "derived"]).all():
        bad = sorted(
            table.loc[
                ~table["kind"].isin(["latent_trait", "item", "outcome", "derived"]), "kind"
            ].unique()
        )
        raise ValueError(f"Registry {path} has unknown kind values: {bad}")

    return Registry(table=table)
