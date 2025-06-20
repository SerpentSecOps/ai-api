import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

@dataclass
class RequestMetrics:
    request_id: str
    timestamp: float
    model_path: str
    input_tokens: int
    output_tokens: int
    processing_time: float
    queue_time: float = 0.0
    batch_size: int = 1
    error: Optional[str] = None

@dataclass
class PerformanceStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_processed: int = 0
    total_processing_time: float = 0.0
    average_tokens_per_second: float = 0.0
    average_response_time: float = 0.0
    current_queue_size: int = 0
    peak_queue_size: int = 0

class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.stats = PerformanceStats()
        self.lock = threading.Lock()
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        
    def start_request(self, request_id: str) -> float:
        """Mark the start of a request"""
        start_time = time.time()
        with self.lock:
            self.active_requests[request_id] = start_time
        return start_time
    
    def record_metrics(self, metrics: RequestMetrics):
        """Record metrics for a completed request"""
        with self.lock:
            self.metrics_history.append(metrics)
            self._update_stats(metrics)
            
            # Remove from active requests
            if metrics.request_id in self.active_requests:
                del self.active_requests[metrics.request_id]
    
    def _update_stats(self, metrics: RequestMetrics):
        """Update aggregate statistics"""
        self.stats.total_requests += 1
        
        if metrics.error is None:
            self.stats.successful_requests += 1
            self.stats.total_tokens_processed += metrics.input_tokens + metrics.output_tokens
            self.stats.total_processing_time += metrics.processing_time
            
            # Calculate running averages
            if self.stats.successful_requests > 0:
                self.stats.average_response_time = (
                    self.stats.total_processing_time / self.stats.successful_requests
                )
                
                if self.stats.total_processing_time > 0:
                    self.stats.average_tokens_per_second = (
                        self.stats.total_tokens_processed / self.stats.total_processing_time
                    )
        else:
            self.stats.failed_requests += 1
    
    def update_queue_size(self, current_size: int):
        """Update current queue size and track peak"""
        with self.lock:
            self.stats.current_queue_size = current_size
            if current_size > self.stats.peak_queue_size:
                self.stats.peak_queue_size = current_size
    
    def get_stats(self) -> dict:
        """Get current performance statistics"""
        with self.lock:
            return {
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'success_rate': (
                    self.stats.successful_requests / max(self.stats.total_requests, 1) * 100
                ),
                'total_tokens_processed': self.stats.total_tokens_processed,
                'average_tokens_per_second': round(self.stats.average_tokens_per_second, 2),
                'average_response_time': round(self.stats.average_response_time, 3),
                'current_queue_size': self.stats.current_queue_size,
                'peak_queue_size': self.stats.peak_queue_size,
                'active_requests': len(self.active_requests)
            }
    
    def get_recent_metrics(self, limit: int = 50) -> List[dict]:
        """Get recent request metrics"""
        with self.lock:
            recent = list(self.metrics_history)[-limit:]
            return [
                {
                    'request_id': m.request_id,
                    'timestamp': m.timestamp,
                    'model_path': m.model_path,
                    'input_tokens': m.input_tokens,
                    'output_tokens': m.output_tokens,
                    'processing_time': round(m.processing_time, 3),
                    'queue_time': round(m.queue_time, 3),
                    'batch_size': m.batch_size,
                    'tokens_per_second': round(
                        (m.input_tokens + m.output_tokens) / max(m.processing_time, 0.001), 2
                    ),
                    'error': m.error
                }
                for m in recent
            ]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        with self.lock:
            data = {
                'stats': self.get_stats(),
                'history': self.get_recent_metrics(len(self.metrics_history))
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

# Global metrics collector instance
metrics_collector = MetricsCollector()