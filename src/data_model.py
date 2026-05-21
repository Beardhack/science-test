from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config_path: Path
    reports_dir: Path

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.reports_dir / "tables"


def load_config(path: str | Path = "config/research_question.yaml") -> dict[str, Any]:
    """Load the research question config.

    PyYAML is used when installed. The bundled Codex runtime used for this kata does
    not include PyYAML, so the checked-in config is JSON-compatible YAML and can be
    parsed with the standard library.
    """
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {config_path} must parse to a mapping.")
    return loaded


def project_paths(
    config_path: str | Path = "config/research_question.yaml",
    reports_dir: str | Path = "reports",
) -> ProjectPaths:
    root = Path.cwd()
    paths = ProjectPaths(root=root, config_path=Path(config_path), reports_dir=Path(reports_dir))
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    return paths


def rq(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["research_question"]


def analysis_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("analysis", {})


def nested_get(mapping: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    cursor: Any = mapping
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def clean_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def first_int_from_text(value: str, default: int | None = None) -> int | None:
    match = re.search(r"(-?\d+)", value)
    if not match:
        return default
    return int(match.group(1))


def age_threshold_from_config(config: Mapping[str, Any], default: int = 40) -> int:
    criteria = nested_get(config, ["research_question", "population", "inclusion_criteria"], [])
    for criterion in criteria:
        if "age" in str(criterion).lower():
            parsed = first_int_from_text(str(criterion))
            if parsed is not None:
                return parsed
    return default


def follow_up_days(config: Mapping[str, Any]) -> int:
    return int(nested_get(config, ["research_question", "outcome", "primary_follow_up_days"], 365))


def baseline_days(config: Mapping[str, Any]) -> int:
    return int(nested_get(config, ["research_question", "baseline", "covariate_assessment_window_days"], 365))


def washout_days(config: Mapping[str, Any]) -> int:
    return int(nested_get(config, ["research_question", "baseline", "washout_window_days"], 365))


def exposure_label(config: Mapping[str, Any]) -> str:
    return str(nested_get(config, ["research_question", "exposure", "name"], "Exposure"))


def comparator_label(config: Mapping[str, Any]) -> str:
    return str(nested_get(config, ["research_question", "comparator", "name"], "Comparator"))


def empty_or_blank(values: Any) -> bool:
    if values is None:
        return True
    if not isinstance(values, list):
        return str(values).strip() == ""
    return len([v for v in values if str(v).strip()]) == 0


def write_markdown(path: str | Path, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content.rstrip() + "\n", encoding="utf-8")
