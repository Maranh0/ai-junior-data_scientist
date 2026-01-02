# metrics_logger.py

import time
from typing import Any, Dict


class MetricsLogger:
    def __init__(self):
        self.total_requests = 0

    def wrap_call(self, agent_executor, message: str) -> Dict[str, Any]:
        """
        Measure latency and wrap an agent call.
        """
        start = time.time()
        result = agent_executor.invoke({"input": message})
        end = time.time()

        self.total_requests += 1

        metrics = {
            "latency_sec": end - start,
            "total_requests": self.total_requests,
        }
        return {"result": result, "metrics": metrics}
