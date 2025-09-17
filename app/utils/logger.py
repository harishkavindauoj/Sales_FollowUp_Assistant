import structlog
import logging
from typing import Any, Dict
import os

# Configure structlog
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

# Set log level from environment
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))

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
        """Log Bedrock API call metrics"""
        self.logger.info(
            "Bedrock API call",
            latency_seconds=latency,
            tokens_used=tokens_used,
            cost_estimate_usd=cost_estimate
        )

    @staticmethod
    def _redact_pii(data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact potential PII from inputs"""
        redacted = {}
        for key, value in data.items():
            if any(pii_field in key.lower() for pii_field in ['email', 'phone', 'name', 'address']):
                redacted[key] = "[REDACTED]"
            elif isinstance(value, (str, int, float, bool)):
                redacted[key] = value
            else:
                redacted[key] = f"<{type(value).__name__}>"
        return redacted