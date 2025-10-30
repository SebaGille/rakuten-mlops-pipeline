# flows/monitor_and_retrain.py
"""Prefect flow that checks data drift then optionally retrains the model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import flow, task, get_run_logger

# Ensure project modules are importable when the flow is executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import check_drift  # type: ignore  # pylint: disable=import-error
from flows.pipeline_flow import rakuten_pipeline  # type: ignore  # pylint: disable=import-error


@task(name="run-check-drift")
def run_check_drift() -> Dict[str, Any]:
    """Execute the drift check script and return its findings."""

    logger = get_run_logger()
    logger.info("Running drift check...")

    exit_code = check_drift.main()

    status_payload: Dict[str, Any] = {}
    status_file = check_drift.STATUS_FILE
    if status_file.exists():
        try:
            status_payload = json.loads(status_file.read_text(encoding="utf-8"))
            logger.info(
                "Drift status: share=%.3f threshold=%.3f trigger=%s",
                status_payload.get("drift_share"),
                status_payload.get("drift_threshold"),
                status_payload.get("trigger_retrain"),
            )
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse drift status JSON (%s)", exc)
    else:
        logger.warning("Drift status file not found at %s", status_file)

    return {"exit_code": exit_code, "status": status_payload}


@task(name="run-training-pipeline")
def run_training_pipeline(run_predict: bool = True, sync_dvc: bool = True) -> Optional[str]:
    """Trigger the full training pipeline flow with optional steps."""

    logger = get_run_logger()
    logger.info(
        "Launching training pipeline (predict=%s, sync_dvc=%s)", run_predict, sync_dvc
    )

    # Prefect flows can be called like regular functions; tasks execute within context
    result = rakuten_pipeline(run_predict=run_predict, sync_dvc=sync_dvc)

    logger.info("Training pipeline completed successfully.")
    return result


@flow(name="monitor-and-retrain")
def monitor_and_retrain(trigger_on_parse_error: bool = True) -> None:
    """Run drift monitoring and optionally retrain when the threshold is exceeded."""

    outcome = run_check_drift()
    status = outcome.get("status", {})
    exit_code = int(outcome.get("exit_code", 1))

    trigger_flag = bool(status.get("trigger_retrain", False))
    logger = get_run_logger()

    if trigger_flag:
        logger.warning("Drift threshold exceeded — triggering retraining.")
        run_training_pipeline()
        return

    if exit_code != 0 and trigger_on_parse_error:
        logger.warning(
            "Drift check returned non-zero exit code (%s); retraining as fail-safe.",
            exit_code,
        )
        run_training_pipeline()
        return

    logger.info("Drift below threshold — no retraining required.")


if __name__ == "__main__":
    monitor_and_retrain()
