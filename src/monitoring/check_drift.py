"""CLI helper to check data drift on inference traffic.

This script shares the same preprocessing primitives as
`src/monitoring/generate_evidently.py` so that both utilities operate on an
identical feature space (`text_len`, `prdtypecode`). The output is a compact
status file consumed by automation (CI/CD, alerts, etc.) indicating whether the
current drift share crossed a configurable threshold.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from evidently import Report, metrics

try:  # pragma: no cover - import shim for direct script execution
    from .generate_evidently import load_reference_df, load_current_df
except ImportError:  # running as `python src/monitoring/check_drift.py`
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
    from monitoring.generate_evidently import load_reference_df, load_current_df


# --- Paths & Config ---
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
REPORT_DIR = PROJECT_ROOT / "reports" / "evidently"
STATUS_FILE = REPORT_DIR / "drift_status.json"
MONITORED_COLUMNS = ("text_len", "prdtypecode")


def _load_threshold(env_var: str = "DRIFT_THRESHOLD", default: str = "0.3") -> float:
    raw_value = os.getenv(env_var, default)
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {env_var}='{raw_value}'. Provide a numeric value between 0 and 1."
        ) from exc


DRIFT_THRESHOLD = _load_threshold()


def _run_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Create and execute an Evidently report for monitored columns."""

    drift_metrics = [metrics.ValueDrift(column=col) for col in MONITORED_COLUMNS]
    report = Report(metrics=[metrics.DriftedColumnsCount(), *drift_metrics, metrics.RowCount(), metrics.ColumnCount()])
    return report.run(reference_data=reference, current_data=current)


def _extract_summary(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Evidently output to obtain drift statistics.

    Expected structure is aligned with what `Report.save_json` produces. The
    function is defensive so the script fails loudly if Evidently changes its
    serialization format.
    """

    drift_share = None
    drifted_columns = None
    column_values: Dict[str, Any] = {}

    for metric_payload in report_dict.get("metrics", []):
        metric_id = metric_payload.get("metric_id") or metric_payload.get("id")
        value = metric_payload.get("value")

        if not isinstance(metric_id, str):
            continue

        if metric_id.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drift_share = value.get("share")
                drifted_columns = value.get("count")
        elif metric_id.startswith("ValueDrift(column="):
            column_name = metric_id.split("=", 1)[1].rstrip(")")
            column_values[column_name] = value

    if drift_share is None:
        raise RuntimeError("Unable to extract drift share from Evidently report output.")

    return {
        "drift_share": drift_share,
        "drifted_columns": drifted_columns,
        "column_values": column_values,
    }


def _report_to_dict(report_result) -> Dict[str, Any]:
    """Convert an Evidently report result to a Python dictionary."""

    if hasattr(report_result, "as_dict"):
        return report_result.as_dict()

    if hasattr(report_result, "json"):
        try:
            return json.loads(report_result.json())
        except (TypeError, json.JSONDecodeError):
            pass

    if hasattr(report_result, "save_json"):
        with tempfile.NamedTemporaryFile(mode="r+", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            report_result.save_json(str(tmp_path))
            return json.loads(tmp_path.read_text(encoding="utf-8"))
        finally:
            tmp_path.unlink(missing_ok=True)

    raise AttributeError(
        "Unable to serialise Evidently report result. Update Evidently or adjust this helper."
    )


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reference = load_reference_df()
    current = load_current_df()

    report_result = _run_report(reference, current)
    summary = _extract_summary(_report_to_dict(report_result))

    drift_share = float(summary["drift_share"])
    trigger_retrain = drift_share >= DRIFT_THRESHOLD

    status = {
        "drift_share": drift_share,
        "drift_threshold": DRIFT_THRESHOLD,
        "drifted_columns": summary["drifted_columns"],
        "monitored_columns": list(MONITORED_COLUMNS),
        "column_values": summary["column_values"],
        "trigger_retrain": trigger_retrain,
    }

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"[INFO] Drift share = {drift_share:.3f} (threshold = {DRIFT_THRESHOLD:.3f})")
    for column, value in summary["column_values"].items():
        if isinstance(value, (int, float)):
            print(f"[INFO]   › {column}: metric value = {value:.4f}")
        else:
            print(f"[INFO]   › {column}: metric payload = {value}")

    if trigger_retrain:
        print("[WARN] Drift threshold exceeded. Retrain trigger set to True.")
    else:
        print("[INFO] Drift within acceptable range. Retrain trigger remains False.")

    print(f"[INFO] Status written to {STATUS_FILE}")

    return 1 if trigger_retrain else 0


if __name__ == "__main__":
    sys.exit(main())
