from typing import Dict, Any, List, Optional
import json
import re
from datetime import datetime, date
import pandas as pd


def sanitize_customer_id(customer_id: str) -> str:
    """Sanitize and validate customer ID"""
    if not customer_id:
        raise ValueError("Customer ID cannot be empty")

    # Remove whitespace and convert to uppercase
    clean_id = customer_id.strip().upper()

    # Basic validation - alphanumeric with possible hyphens/underscores
    if not re.match(r'^[A-Z0-9_-]+$', clean_id):
        raise ValueError(f"Invalid customer ID format: {customer_id}")

    return clean_id


def validate_date_string(date_str: str) -> date:
    """Validate and parse date string in YYYY-MM-DD format"""
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to int with fallback"""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def truncate_string(text: str, max_length: int = 500) -> str:
    """Truncate string to max length with ellipsis"""
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def format_currency(amount: float) -> str:
    """Format amount as currency string"""
    return f"${amount:.2f}"


def calculate_days_between(start_date: datetime, end_date: datetime = None) -> int:
    """Calculate days between dates"""
    if end_date is None:
        end_date = datetime.now()

    return (end_date - start_date).days


def extract_top_items(items_dict: Dict[str, int], top_n: int = 3) -> List[tuple]:
    """Extract top N items from a dictionary by value"""
    return sorted(items_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]


def validate_rfm_score(score: int) -> int:
    """Validate and clamp RFM score to 0-100 range"""
    return max(0, min(100, safe_int_conversion(score)))


def validate_churn_risk(risk: float) -> float:
    """Validate and clamp churn risk to 0-1 range"""
    return max(0.0, min(1.0, safe_float_conversion(risk)))


def validate_priority(priority: int) -> int:
    """Validate and clamp priority to 1-5 range"""
    return max(1, min(5, safe_int_conversion(priority, 1)))


def clean_json_response(response_text: str) -> str:
    """Clean JSON response text by removing markdown and extra formatting"""
    if not response_text:
        return "{}"

    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)

    # Remove any leading/trailing whitespace
    cleaned = cleaned.strip()

    # If it doesn't start with { or [, try to find JSON content
    if not cleaned.startswith(('{', '[')):
        json_match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)

    return cleaned


def validate_json_structure(json_data: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate that JSON data has required keys"""
    if not isinstance(json_data, dict):
        return False

    return all(key in json_data for key in required_keys)


def format_recommendation(action: str, reason: str) -> Dict[str, str]:
    """Format a recommendation with validation"""
    valid_actions = ["call", "email", "offer_bundle", "promo"]

    # Validate action
    action_clean = action.lower().strip()
    if action_clean not in valid_actions:
        action_clean = "email"  # Default fallback

    # Clean and validate reason
    reason_clean = truncate_string(reason.strip(), 200)
    if not reason_clean:
        reason_clean = "Follow up required"

    return {
        "action": action_clean,
        "reason": reason_clean
    }


def handle_pandas_timestamp(timestamp_obj) -> str:
    """Handle pandas Timestamp objects and convert to string"""
    if pd.isna(timestamp_obj):
        return ""

    try:
        # If it's a pandas Timestamp, convert to datetime first
        if hasattr(timestamp_obj, 'to_pydatetime'):
            dt = timestamp_obj.to_pydatetime()
        elif isinstance(timestamp_obj, datetime):
            dt = timestamp_obj
        else:
            # Try to parse as string
            dt = pd.to_datetime(timestamp_obj).to_pydatetime()

        return dt.strftime('%Y-%m-%d')
    except Exception:
        return str(timestamp_obj)


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count (1 token ≈ 4 characters)"""
    return len(text) // 4 if text else 0


def calculate_cost_estimate(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost estimate for Bedrock Nova calls"""
    # Nova pricing (approximate)
    INPUT_COST_PER_1K = 0.0008  # $0.0008 per 1K input tokens
    OUTPUT_COST_PER_1K = 0.0032  # $0.0032 per 1K output tokens

    input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K

    return round(input_cost + output_cost, 6)


def create_error_response(customer_id: str, error_message: str) -> Dict[str, Any]:
    """Create a fallback error response that matches the required schema"""
    return {
        "customer_id": customer_id,
        "scores": {
            "rfm_score": 0,
            "churn_risk": 1.0,
            "priority": 1
        },
        "summary": f"Analysis failed: {truncate_string(error_message, 200)}",
        "recommendations": [
            {
                "action": "email",
                "reason": "Manual review required due to system error"
            }
        ],
        "top_followups_today": []
    }


def redact_sensitive_data(data: Any, sensitive_keys: List[str] = None) -> Any:
    """Redact sensitive data from logs"""
    if sensitive_keys is None:
        sensitive_keys = ['email', 'phone', 'address', 'name', 'credit_card']

    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_sensitive_data(value, sensitive_keys)
        return redacted
    elif isinstance(data, list):
        return [redact_sensitive_data(item, sensitive_keys) for item in data]
    else:
        return data