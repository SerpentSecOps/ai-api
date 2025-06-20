import json
import time
import uuid
import threading
from queue import Queue, Empty
from http.server import BaseHTTPRequestHandler
from typing import Dict, List, Any
from core.metrics import metrics_collector, RequestMetrics
from core.model_loader import BatchRequest

class BatchProcessor:
    def __init__(self, max_batch_size: int = 10, timeout: float = 30.0):
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.request_queue = Queue()
        self.result_queues: Dict[str, Queue] = {}
        self.processing = False
        self.worker_thread = None
        self.model_loader = None
        
    def set_model_loader(self, model_loader):
        self.model_loader = model_loader
        
    def start_processing(self):
        if not self.processing:
            self.processing = True
            self.worker_thread = threading.Thread(target=self._process_batches, daemon=True)
            self.worker_thread.start()
    
    def stop_processing(self):
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def add_request(self, request_id: str, messages: List[Dict], settings: Dict) -> Queue:
        result_queue = Queue()
        self.result_queues[request_id] = result_queue
        
        batch_request = BatchRequest(
            request_id=request_id,
            messages=messages,
            settings=settings,
            timestamp=time.time()
        )
        
        self.request_queue.put(batch_request)
        metrics_collector.update_queue_size(self.request_queue.qsize())
        
        return result_queue
    
    def _process_batches(self):
        while self.processing:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    time.sleep(0.1)  # Short sleep if no requests
            except Exception as e:
                print(f"Error in batch processing: {e}")
    
    def _collect_batch(self) -> List[BatchRequest]:
        batch = []
        start_time = time.time()
        
        # Wait for first request
        try:
            first_request = self.request_queue.get(timeout=1.0)
            batch.append(first_request)
        except Empty:
            return batch
        
        # Collect additional requests up to batch size or timeout
        while (len(batch) < self.max_batch_size and 
               time.time() - start_time < self.timeout):
            try:
                request = self.request_queue.get(timeout=0.1)
                batch.append(request)
            except Empty:
                break
        
        metrics_collector.update_queue_size(self.request_queue.qsize())
        return batch
    
    def _process_batch(self, batch: List[BatchRequest]):
        if not self.model_loader:
            # Return errors for all requests
            for req in batch:
                if req.request_id in self.result_queues:
                    self.result_queues[req.request_id].put({
                        "error": "Model not loaded",
                        "content": None
                    })
            return
        
        try:
            results = self.model_loader.create_batch_completion(batch)
            
            # Send results to appropriate queues
            for result in results:
                request_id = result["request_id"]
                if request_id in self.result_queues:
                    self.result_queues[request_id].put(result)
                    
                    # Record metrics
                    req = next((r for r in batch if r.request_id == request_id), None)
                    if req:
                        input_tokens = self.model_loader.count_tokens(
                            str(req.messages)
                        )
                        output_tokens = self.model_loader.count_tokens(
                            result.get("content", "")
                        ) if result.get("content") else 0
                        
                        metrics = RequestMetrics(
                            request_id=request_id,
                            timestamp=time.time(),
                            model_path=self.model_loader.model_path,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            processing_time=result.get("processing_time", 0),
                            queue_time=time.time() - req.timestamp,
                            batch_size=len(batch),
                            error=result.get("error")
                        )
                        metrics_collector.record_metrics(metrics)
                        
        except Exception as e:
            # Return error for all requests in batch
            for req in batch:
                if req.request_id in self.result_queues:
                    self.result_queues[req.request_id].put({
                        "error": str(e),
                        "content": None
                    })

class APIRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, auth_manager=None, batch_processor=None, **kwargs):
        self.auth_manager = auth_manager
        self.batch_processor = batch_processor
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self._send_cors_headers()
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        self._send_cors_headers()
        
        if self.path == '/v1/metrics':
            self._handle_metrics_request()
        elif self.path == '/v1/health':
            self._handle_health_request()
        else:
            self._send_response(404, {"error": "Not found"})
    
    def do_POST(self):
        """Handle POST requests"""
        self._send_cors_headers()
        
        # Authentication check
        if self.auth_manager and not self._authenticate():
            return
        
        if self.path == '/v1/chat/completions':
            self._handle_chat_completion()
        elif self.path == '/v1/chat/batch':
            self._handle_batch_completion()
        else:
            self._send_response(404, {"error": "Not found"})
    
    def _authenticate(self) -> bool:
        """Authenticate the request"""
        if not self.auth_manager.authenticate_request(dict(self.headers)):
            self._send_response(401, {"error": "Authentication required"}, {
                'WWW-Authenticate': 'Basic realm="LLM Server"'
            })
            return False
        return True
    
    def _send_cors_headers(self):
        """Send CORS headers"""
        # Get CORS origins from app state or default
        cors_origins = getattr(self.server, 'cors_origins', ['*'])
        origin = self.headers.get('Origin', '')
        
        if '*' in cors_origins or origin in cors_origins:
            self.send_header('Access-Control-Allow-Origin', origin or '*')
        
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')
    
    def _handle_chat_completion(self):
        """Handle single chat completion request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            
            request_id = str(uuid.uuid4())
            start_time = metrics_collector.start_request(request_id)
            
            # Get model loader from server
            model_loader = getattr(self.server, 'model_loader', None)
            if not model_loader:
                self._send_response(503, {"error": "Model is not loaded"})
                return
            
            # Get settings from server
            settings = getattr(self.server, 'model_settings', {})
            
            if self.batch_processor:
                # Use batch processor
                result_queue = self.batch_processor.add_request(
                    request_id, request_body['messages'], settings
                )
                
                # Wait for result
                try:
                    result = result_queue.get(timeout=60.0)  # 60 second timeout
                    if request_id in self.batch_processor.result_queues:
                        del self.batch_processor.result_queues[request_id]
                    
                    if result.get('error'):
                        self._send_response(500, {"error": result['error']})
                        return
                    
                    generated_text = result['content']
                    
                except Empty:
                    self._send_response(408, {"error": "Request timeout"})
                    return
            else:
                # Direct processing
                try:
                    generated_text = model_loader.create_completion(request_body, settings)
                except Exception as e:
                    self._send_response(500, {"error": f"Generation error: {str(e)}"})
                    return
            
            # Build response
            response_data = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": os.path.basename(model_loader.model_path),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": model_loader.count_tokens(str(request_body['messages'])),
                    "completion_tokens": model_loader.count_tokens(generated_text),
                    "total_tokens": model_loader.count_tokens(str(request_body['messages'])) + model_loader.count_tokens(generated_text)
                }
            }
            
            self._send_response(200, response_data)
            
        except json.JSONDecodeError:
            self._send_response(400, {"error": "Invalid JSON"})
        except Exception as e:
            self._send_response(500, {"error": f"Internal server error: {str(e)}"})
    
    def _handle_batch_completion(self):
        """Handle batch completion request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            
            requests = request_body.get('requests', [])
            if not requests:
                self._send_response(400, {"error": "No requests provided"})
                return
            
            # Limit batch size
            max_batch = getattr(self.server, 'max_batch_size', 10)
            if len(requests) > max_batch:
                self._send_response(400, {
                    "error": f"Batch size exceeds maximum of {max_batch}"
                })
                return
            
            model_loader = getattr(self.server, 'model_loader', None)
            if not model_loader:
                self._send_response(503, {"error": "Model is not loaded"})
                return
            
            settings = getattr(self.server, 'model_settings', {})
            
            # Process batch
            batch_requests = []
            for i, req in enumerate(requests):
                batch_requests.append(BatchRequest(
                    request_id=f"batch-{i}",
                    messages=req.get('messages', []),
                    settings=settings,
                    timestamp=time.time()
                ))
            
            results = model_loader.create_batch_completion(batch_requests)
            
            # Format response
            response_data = {
                "object": "batch.completion",
                "created": int(time.time()),
                "results": [
                    {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": result['content']
                        } if result['content'] else None,
                        "error": result.get('error'),
                        "processing_time": result.get('processing_time', 0)
                    }
                    for i, result in enumerate(results)
                ]
            }
            
            self._send_response(200, response_data)
            
        except json.JSONDecodeError:
            self._send_response(400, {"error": "Invalid JSON"})
        except Exception as e:
            self._send_response(500, {"error": f"Internal server error: {str(e)}"})
    
    def _handle_metrics_request(self):
        """Handle metrics request"""
        metrics_data = {
            "performance": metrics_collector.get_stats(),
            "recent_requests": metrics_collector.get_recent_metrics(20)
        }
        self._send_response(200, metrics_data)
    
    def _handle_health_request(self):
        """Handle health check request"""
        model_loader = getattr(self.server, 'model_loader', None)
        health_data = {
            "status": "healthy" if model_loader and model_loader.is_loaded else "unhealthy",
            "model_loaded": model_loader is not None and model_loader.is_loaded,
            "model_path": model_loader.model_path if model_loader else None,
            "timestamp": time.time()
        }
        self._send_response(200, health_data)
    
    def _send_response(self, status_code: int, content: Dict, extra_headers: Dict = None):
        """Send JSON response with CORS headers"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self._send_cors_headers()
        
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        
        self.end_headers()
        self.wfile.write(json.dumps(content).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        return