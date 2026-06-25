import os
import json
import time

class TelemetryReporter:
    """
    Writes telemetry updates to a dedicated JSON file for cross-process IPC.
    Using individual files per metric_id avoids all multiprocessing lock contention.
    """
    def __init__(self, metric_id: str):
        self.metric_id = metric_id
        self.filepath = os.path.abspath(f"artifacts/telemetry_{metric_id}.json")
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def update(self, current: float, total: float, label: str):
        """
        Atomically updates the telemetry state.
        """
        data = {
            "metric_id": self.metric_id,
            "current": float(current),
            "total": float(total),
            "label": str(label),
            "timestamp": time.time()
        }
        
        # Write to temp file then rename for atomic swap
        temp_path = self.filepath + ".tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f)
            os.replace(temp_path, self.filepath)
        except OSError:
            pass # On Windows, if file is locked for read, it's fine to skip one frame

    def clear(self):
        """Removes the telemetry file when complete."""
        try:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
        except OSError:
            pass
