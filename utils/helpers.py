"""
Utility functions and helpers for the LLM server application.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List

def validate_port(port: str) -> bool:
    """Validate port number"""
    try:
        port_num = int(port)
        return 1024 <= port_num <= 65535
    except ValueError:
        return False

def validate_positive_int(value: str) -> bool:
    """Validate positive integer"""
    try:
        return int(value) > 0
    except ValueError:
        return False

def validate_float_range(value: str, min_val: float, max_val: float) -> bool:
    """Validate float within range"""
    try:
        val = float(value)
        return min_val <= val <= max_val
    except ValueError:
        return False

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely load JSON with fallback"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def ensure_directory(path: str) -> bool:
    """Ensure directory exists, create if not"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError:
        return False

class RateLimiter:
    """Simple rate limiter"""
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False