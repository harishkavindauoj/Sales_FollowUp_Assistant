import structlog
import logging
from typing import Any, Dict
import os
from datetime import datetime

# Configure logging to file instead of terminal
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create log file with timestamp
log_filename = f"{log_dir}/sales_assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure file logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Keep minimal console output
    ]
)

# Configure structlog with file output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class NodeLogger:
    """Logger for graph nodes with standardized structure"""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.logger = logger.bind(node=node_name)

    def log_start(self, inputs: Dict[str, Any]):
        """Log node start with redacted inputs"""
        safe_inputs = self._redact_pii(inputs)
        self.logger.info("Node started", inputs_keys=list(safe_inputs.keys()))

    def log_end(self, outputs: Dict[str, Any], success: bool = True):
        """Log node completion with output size only"""
        output_info = {
            key: len(str(value)) if value else 0
            for key, value in outputs.items()
        }
        self.logger.info(
            "Node completed",
            success=success,
            output_sizes=output_info
        )

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log node error"""
        self.logger.error(
            "Node error",
            error=str(error),
            error_type=type(error).__name__,
            context=context or {}
        )

    def log_bedrock_call(self, latency: float, tokens_used: int, cost_estimate: float):
        """Log Bedrock API call metrics - REQUIREMENT 6"""
        self.logger.info(
            "Bedrock API call metrics",
            latency_seconds=round(latency, 3),
            tokens_used=tokens_used,
            cost_estimate_usd=round(cost_estimate, 6),
            node=self.node_name
        )

        # Also log to console for immediate visibility
        print(f"💰 Bedrock Call | Node: {self.node_name} | "
              f"Latency: {latency:.3f}s | Tokens: {tokens_used} | "
              f"Cost: ${cost_estimate:.6f}")

    def log_timeout_warning(self, latency: float, timeout_threshold: float):
        """Log circuit breaker timeout warning - REQUIREMENT 6"""
        self.logger.warning(
            "Circuit breaker timeout warning",
            latency_seconds=round(latency, 3),
            timeout_threshold=timeout_threshold,
            node=self.node_name
        )

        print(f"⚠️ TIMEOUT WARNING | Node: {self.node_name} | "
              f"Latency: {latency:.3f}s exceeded {timeout_threshold}s threshold")

    @staticmethod
    def _redact_pii(data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact potential PII from inputs - REQUIREMENT 7"""
        redacted = {}
        pii_fields = ['email', 'phone', 'name', 'address', 'credit_card', 'ssn']

        for key, value in data.items():
            # Check if key contains PII indicators
            if any(pii_field in key.lower() for pii_field in pii_fields):
                redacted[key] = "[REDACTED]"
            elif isinstance(value, (str, int, float, bool)):
                redacted[key] = value
            elif isinstance(value, dict):
                redacted[key] = NodeLogger._redact_pii(value)
            else:
                redacted[key] = f"<{type(value).__name__}>"
        return redacted


# Console logger for immediate feedback
class ConsoleLogger:
    """Simple console logger for important events"""

    @staticmethod
    def log_analysis_start(customer_id: str):
        print(f"🔍 Starting analysis for customer: {customer_id}")

    @staticmethod
    def log_analysis_complete(customer_id: str, duration: float):
        print(f"✅ Analysis complete for {customer_id} in {duration:.2f}s")

    @staticmethod
    def log_analysis_error(customer_id: str, error: str):
        print(f"❌ Analysis failed for {customer_id}: {error}")

    @staticmethod
    def log_parallel_execution(branches: list):
        print(f"🔀 Executing parallel branches: {', '.join(branches)}")


# Export logger instance
__all__ = ['logger', 'NodeLogger', 'ConsoleLogger']