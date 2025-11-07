"""Utilities to manage AWS ECS services from Streamlit."""

from __future__ import annotations

import os
from typing import Dict, Optional, Any

import boto3
from botocore.exceptions import ClientError


DEFAULT_REGION = os.getenv("AWS_REGION", "eu-west-1")
DEFAULT_CLUSTER = os.getenv("AWS_ECS_CLUSTER", "rakuten-mlops-cluster")
DEFAULT_API_SERVICE = os.getenv("ECS_API_SERVICE_NAME", "rakuten-api-service")
DEFAULT_MLFLOW_SERVICE = os.getenv("ECS_MLFLOW_SERVICE_NAME", "rakuten-mlflow-service")
DEFAULT_RDS_INSTANCE = os.getenv("RDS_INSTANCE_ID", "rakuten-mlflow-db")


class ECSManager:
    """Helper around boto3 ECS calls for the Streamlit dashboard."""

    def __init__(
        self,
        region: str = DEFAULT_REGION,
        cluster: str = DEFAULT_CLUSTER,
        services: Optional[Dict[str, Dict[str, Any]]] = None,
        rds_instance_id: Optional[str] = DEFAULT_RDS_INSTANCE,
    ) -> None:
        self.region = region
        self.cluster = cluster
        self.ecs = boto3.client("ecs", region_name=self.region)
        self.elbv2 = boto3.client("elbv2", region_name=self.region)
        self.rds = (
            boto3.client("rds", region_name=self.region)
            if rds_instance_id
            else None
        )
        self.rds_instance_id = rds_instance_id

        alb_url = os.getenv("AWS_ALB_URL", "").rstrip("/")
        # Use ALB URL directly - ALB routes based on Host header, not path prefix
        default_api_url = os.getenv("API_URL")
        if not default_api_url and alb_url:
            default_api_url = alb_url  # No /api prefix - routing is via Host header
        default_mlflow_url = os.getenv("MLFLOW_URL")
        if not default_mlflow_url and alb_url:
            default_mlflow_url = alb_url  # No /mlflow prefix - routing is via Host header

        self.services = services or {
            "api": {
                "serviceName": DEFAULT_API_SERVICE,
                "displayName": "Rakuten API",
                "url": default_api_url,
            },
            "mlflow": {
                "serviceName": DEFAULT_MLFLOW_SERVICE,
                "displayName": "MLflow Server",
                "url": default_mlflow_url,
            },
        }

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Return status information for each configured ECS service."""

        response = self.ecs.describe_services(
            cluster=self.cluster,
            services=[cfg["serviceName"] for cfg in self.services.values()],
            include=["TAGS"],
        )

        statuses: Dict[str, Dict[str, Any]] = {}

        service_map = {
            svc["serviceName"]: svc for svc in response.get("services", [])
        }

        for key, cfg in self.services.items():
            svc = service_map.get(cfg["serviceName"])
            if not svc:
                statuses[key] = {
                    "exists": False,
                    "running": False,
                    "desired": 0,
                    "status": "MISSING",
                    "health": "unknown",
                    "displayName": cfg.get("displayName", key.title()),
                    "url": cfg.get("url"),
                }
                continue

            running = svc.get("runningCount", 0)
            desired = svc.get("desiredCount", 0)
            deployment = (svc.get("deployments") or [{}])[0]
            rollout_state = deployment.get("rolloutState")

            if running == 0:
                health = "stopped"
            elif running == desired and rollout_state == "COMPLETED":
                health = "healthy"
            else:
                health = "deploying"

            latest_event = None
            events = svc.get("events") or []
            if events:
                latest_event = events[0].get("message")

            statuses[key] = {
                "exists": True,
                "running": running > 0,
                "runningCount": running,
                "desired": desired,
                "status": svc.get("status"),
                "health": health,
                "displayName": cfg.get("displayName", key.title()),
                "url": cfg.get("url"),
                "lastEvent": latest_event,
                "serviceArn": svc.get("serviceArn"),
            }

        return statuses

    def get_rds_status(self) -> Optional[Dict[str, Any]]:
        """Return status information for the linked RDS instance (if configured)."""

        if not self.rds or not self.rds_instance_id:
            return None

        try:
            resp = self.rds.describe_db_instances(
                DBInstanceIdentifier=self.rds_instance_id
            )
        except ClientError as exc:  # pragma: no cover - informative path
            return {"status": f"error: {exc.response['Error']['Message']}"}

        db = resp.get("DBInstances", [{}])[0]
        return {
            "status": db.get("DBInstanceStatus"),
            "endpoint": db.get("Endpoint", {}).get("Address"),
            "engine": db.get("Engine"),
            "multiAZ": db.get("MultiAZ"),
        }

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------
    def scale_service(self, key: str, desired_count: int) -> None:
        """Update the desired count for a specific service."""

        cfg = self.services.get(key)
        if not cfg:
            raise ValueError(f"Unknown service key: {key}")

        self.ecs.update_service(
            cluster=self.cluster,
            service=cfg["serviceName"],
            desiredCount=desired_count,
        )

    def scale_all(self, desired_count: int) -> None:
        """Scale every managed service to `desired_count`."""

        for key in self.services:
            self.scale_service(key, desired_count)


