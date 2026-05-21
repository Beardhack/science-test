from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .data_model import follow_up_days, nested_get


def validate_outcome_columns(df: pd.DataFrame, config: Mapping[str, Any]) -> None:
    followup = follow_up_days(config)
    required = ["event_365", "followup_days", "time_zero"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required outcome columns: {missing}")
    if (df["followup_days"] <= 0).any():
        raise ValueError("Follow-up time must be positive.")
    if (df["followup_days"] > followup).any():
        raise ValueError("Treatment-policy follow-up exceeds configured primary follow-up.")
    if not set(df["event_365"].dropna().unique()).issubset({0, 1}):
        raise ValueError("event_365 must be binary.")


def outcome_definition_markdown(config: Mapping[str, Any]) -> str:
    outcome = nested_get(config, ["research_question", "outcome"], {})
    return (
        f"Primary outcome: {outcome.get('name', 'configured outcome')}.\n\n"
        f"Type: {outcome.get('outcome_type', 'unspecified')}.\n\n"
        f"Primary follow-up: {outcome.get('primary_follow_up_days', 365)} days.\n\n"
        f"Ascertainment notes: {outcome.get('outcome_ascertainment_notes', 'None supplied.')}"
    )
