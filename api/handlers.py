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
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in batch processing: {e}")
    
    def _collect_batch(self) -> List[BatchRequest]:
        batch = []
        start_time = time.time()
        
        try:
            first_request = self.request_queue.get(timeout=1.0)
            batch.append(first_request)
        except Empty:
            return batch
        
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
            for req in batch:
                if req.request_id in self.result_queues:
                    self.result_queues[req.request_id].put({"error": "Model not loaded", "content": None})
            return
        
        try:
            results = self.model_loader.create_batch_completion(batch)
            for result in results:
                request_id = result["request_id"]
                if request_id in self.result_queues:
                    self.result_queues[request_id].put(result)
                    req = next((r for r in batch if r.request_id == request_id), None)
                    if req:
                        metrics = RequestMetrics(
                            request_id=request_id,
                            timestamp=time.time(),
                            model_path=self.model_loader.model_path,
                            input_tokens=self.model_loader.count_tokens(str(req.messages)),
                            output_tokens=self.model_loader.count_tokens(result.get("content", "") or ""),
                            processing_time=result.get("processing_time", 0),
                            queue_time=time.time() - req.timestamp,
                            batch_size=len(batch),
                            error=result.get("error")
                        )
                        metrics_collector.record_metrics(metrics)
                        
        except Exception as e:
            for req in batch:
                if req.request_id in self.result_queues:
                    self.result_queues[req.request_id].put({"error": str(e), "content": None})

class APIRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, auth_manager=None, batch_processor=None, **kwargs):
        self.auth_manager = auth_manager
        self.batch_processor = batch_processor
        super().__init__(*args, **kwargs)

    def _send_cors_headers(self):
        """Handle sending of CORS headers for non-preflight requests."""
        cors_origins = getattr(self.server, 'cors_origins', [])
        origin = self.headers.get('Origin')

        if '*' in cors_origins:
            self.send_header('Access-Control-Allow-Origin', '*')
        elif origin and origin in cors_origins:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        print("--- Handling OPTIONS preflight request ---")
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def _handle_request(self, handler_func):
        """A generic request handler to wrap common logic."""
        status_code = 500
        response_content = {"error": "An unknown internal server error occurred."}
        try:
            if self.command == 'POST':
                if self.auth_manager and not self.auth_manager.authenticate_request(dict(self.headers)):
                    self._send_response(401, {"error": "Authentication required"}, {'WWW-Authenticate': 'Basic realm="LLM Server"'})
                    return

            status_code, response_content = handler_func()

        except json.JSONDecodeError:
            status_code = 400
            response_content = {"error": "Invalid JSON in request body."}
        except Exception as e:
            print(f"ERROR processing {self.path}: {e}")
            if status_code == 500:
                 response_content = {"error": f"Internal server error: {str(e)}"}
            else:
                 response_content['error'] = str(e)

        self._send_response(status_code, response_content)

    def do_GET(self):
        if self.path == '/v1/metrics':
            self._handle_request(self._get_metrics)
        elif self.path == '/v1/health':
            self._handle_request(self._get_health)
        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == '/v1/chat/completions':
            self._handle_request(self._post_chat_completion)
        else:
            self._send_response(404, {"error": "Not found"})
            
    # --- Handler implementations ---

    def _get_metrics(self):
        return 200, {
            "performance": metrics_collector.get_stats(),
            "recent_requests": metrics_collector.get_recent_metrics(20)
        }

    def _get_health(self):
        model_loader = getattr(self.server, 'model_loader', None)
        is_ready = model_loader and model_loader.is_loaded
        return 200, {
            "status": "healthy" if is_ready else "unhealthy",
            "model_loaded": is_ready,
            "model_path": getattr(model_loader, 'model_path', None),
            "timestamp": time.time()
        }
        
    def _post_chat_completion(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data.decode('utf-8'))
        
        request_id = str(uuid.uuid4())
        metrics_collector.start_request(request_id)
        
        model_loader = getattr(self.server, 'model_loader', None)
        if not model_loader or not model_loader.is_loaded:
            return 503, {"error": "Model is not loaded or ready."}
        
        settings = getattr(self.server, 'model_settings', {})
        
        result_queue = self.batch_processor.add_request(
            request_id, request_body['messages'], settings
        )
        try:
            result = result_queue.get(timeout=120.0)
            if request_id in self.batch_processor.result_queues:
                del self.batch_processor.result_queues[request_id]
            
            if result.get('error'):
                raise Exception(result['error'])
            
            generated_text = result['content']
        except Empty:
            return 408, {"error": "Request timed out waiting for model processing."}

        response_data = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_loader.model_path.split('/')[-1],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": model_loader.count_tokens(str(request_body['messages'])),
                "completion_tokens": model_loader.count_tokens(generated_text or ""),
                "total_tokens": model_loader.count_tokens(str(request_body['messages'])) + model_loader.count_tokens(generated_text or "")
            }
        }
        return 200, response_data

    def _send_response(self, status_code, content, headers=None):
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            
            if headers:
                for key, value in headers.items():
                    self.send_header(key, value)
            
            body = json.dumps(content).encode('utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            print(f"CRITICAL: Failed to send HTTP response. Error: {e}")

    def log_message(self, format_str, *args):
        """Re-enable logging to see incoming requests for debugging."""
        print(f"--- Received Request ---")
        super().log_message(format_str, *args)
