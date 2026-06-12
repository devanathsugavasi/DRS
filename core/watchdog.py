"""Background watchdog that monitors DRS subsystem health."""

from __future__ import annotations

import threading
import time
from typing import Any

from utils.logger import get_logger

log = get_logger("watchdog")


class DRSWatchdog:
    """Monitors camera threads, API health, and resource usage."""

    def __init__(self, camera_manager: Any = None, check_interval: float = 10.0) -> None:
        self.camera_manager = camera_manager
        self.check_interval = check_interval
        self._thread: threading.Thread | None = None
        self._running = False
        self.last_report: dict[str, Any] = {}

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="drs-watchdog")
        self._thread.start()
        log.info("Watchdog started with {}s check interval", self.check_interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("Watchdog stopped")

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                report = self._check_health()
                self.last_report = report
                dead = report.get("cameras_dead", [])
                if dead:
                    log.warning("Dead cameras detected: {}", dead)
                    self._restart_dead_cameras(dead)
            except Exception as exc:
                log.error("Watchdog check failed: {}", exc)
            time.sleep(self.check_interval)

    def _check_health(self) -> dict[str, Any]:
        report: dict[str, Any] = {"timestamp": time.time()}

        # Camera health
        if self.camera_manager and hasattr(self.camera_manager, "health"):
            health = self.camera_manager.health()
            report["camera_health"] = health
            report["cameras_dead"] = [
                cam_id for cam_id, info in health.items() if not info.get("alive", True)
            ]
        else:
            report["camera_health"] = {}
            report["cameras_dead"] = []

        # System resources
        try:
            import psutil

            report["memory_percent"] = psutil.virtual_memory().percent
            report["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        except ImportError:
            report["memory_percent"] = None
            report["cpu_percent"] = None

        # Disk space for data directory
        try:
            from pathlib import Path

            data_dir = Path("data")
            if data_dir.exists():
                import shutil

                usage = shutil.disk_usage(data_dir)
                report["storage"] = {
                    "total_gb": round(usage.total / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "used_percent": round(usage.used / usage.total * 100, 1),
                }
        except Exception:
            pass

        return report

    def _restart_dead_cameras(self, dead_ids: list[int]) -> None:
        if not self.camera_manager:
            return
        for cam_id in dead_ids:
            log.info("Attempting to restart camera {}", cam_id)
            try:
                if hasattr(self.camera_manager, "restart_camera"):
                    self.camera_manager.restart_camera(cam_id)
                    log.info("Camera {} restart initiated", cam_id)
            except Exception as exc:
                log.error("Failed to restart camera {}: {}", cam_id, exc)

    def get_report(self) -> dict[str, Any]:
        """Return the latest health report."""
        return dict(self.last_report)
