"""Load and validate the YAML configs into typed contracts (TZ §0).

Nothing that belongs in a config is hardcoded: mains frequency, montages,
thresholds, norms, the rule base, and ACNS terminology all come from
``configs/`` and are validated against the pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..contracts.config import (
    FilterConfig,
    MontageConfig,
    RealtimeConfig,
    RuleBase,
    Thresholds,
)


def configs_dir(explicit: str | Path | None = None) -> Path:
    """Locate the repo ``configs/`` directory."""
    if explicit is not None:
        return Path(explicit)
    # this file: src/neurolens/pipeline/config_loader.py -> repo_root/configs
    return Path(__file__).resolve().parents[3] / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


@dataclass
class ConfigBundle:
    """All runtime configuration, validated."""

    filters: FilterConfig
    thresholds: Thresholds
    montages: dict[str, MontageConfig]
    rule_base: RuleBase
    norms: dict[str, Any]
    acns: dict[str, Any]
    electrode_coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    root: Path | None = None

    def montage(self, name: str) -> MontageConfig:
        if name not in self.montages:
            raise KeyError(
                f"unknown montage {name!r}; available: {sorted(self.montages)}"
            )
        return self.montages[name]


def load_configs(root: str | Path | None = None) -> ConfigBundle:
    """Read every config file and return a validated :class:`ConfigBundle`."""
    cfg = configs_dir(root)

    filters = FilterConfig(**_read_yaml(cfg / "filters.yaml"))
    thresholds = Thresholds(**_read_yaml(cfg / "thresholds.yaml"))
    rule_base = RuleBase(**_read_yaml(cfg / "rules" / "neuro_rules.yaml"))

    montages: dict[str, MontageConfig] = {}
    for mfile in sorted((cfg / "montages").glob("*.yaml")):
        mc = MontageConfig(**_read_yaml(mfile))
        montages[mc.name] = mc

    norms = _read_yaml(cfg / "norms" / "age_norms.yaml")
    acns = _read_yaml(cfg / "acns.yaml")

    coords_raw = _read_yaml(cfg / "electrodes" / "coords_10_20.yaml").get("positions", {})
    coords = {k: (float(v[0]), float(v[1])) for k, v in coords_raw.items()}

    rt_file = cfg / "realtime.yaml"
    realtime = RealtimeConfig(**_read_yaml(rt_file)) if rt_file.exists() else RealtimeConfig()

    return ConfigBundle(
        filters=filters,
        thresholds=thresholds,
        montages=montages,
        rule_base=rule_base,
        norms=norms,
        acns=acns,
        electrode_coords=coords,
        realtime=realtime,
        root=cfg,
    )
