import threading
import socket
from http.server import HTTPServer
from functools import partial
from core.auth import AuthManager
from api.handlers import APIRequestHandler, BatchProcessor
from config.settings import ServerConfig

class LLMServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.auth_manager = AuthManager(
            config.auth_username, 
            config.auth_password
        ) if config.auth_enabled else None
        
        self.batch_processor = BatchProcessor(
            config.batch_max_size,
            config.batch_timeout
        )
        
        self.http_server = None
        self.server_thread = None
        self.model_loader = None
        self.model_settings = {}
        
    def _find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.config.host, port))
                    return port
            except OSError:
                continue
        raise OSError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")
        
    def set_model_loader(self, model_loader, settings):
        """Set the model loader and settings"""
        self.model_loader = model_loader
        self.model_settings = settings
        self.batch_processor.set_model_loader(model_loader)
        
    def start(self):
        """Start the HTTP server"""
        if self.http_server:
            return False  # Already running
        
        # Try to find an available port
        try:
            available_port = self._find_available_port(self.config.port)
            if available_port != self.config.port:
                print(f"Port {self.config.port} is busy, using port {available_port} instead")
                self.config.port = available_port
        except OSError as e:
            print(f"Could not find available port: {e}")
            return False
        
        # Create handler with dependencies
        handler = partial(
            APIRequestHandler,
            auth_manager=self.auth_manager,
            batch_processor=self.batch_processor
        )
        
        # Create server
        server_address = (self.config.host, self.config.port)
        try:
            self.http_server = HTTPServer(server_address, handler)
        except OSError as e:
            if "Permission denied" in str(e) or "Access is denied" in str(e):
                print(f"Permission denied on port {self.config.port}. Try running as administrator or use a port > 1024")
            raise e
        
        # Set server attributes for handlers to access
        self.http_server.model_loader = self.model_loader
        self.http_server.model_settings = self.model_settings
        self.http_server.cors_origins = self.config.cors_origins
        self.http_server.max_batch_size = self.config.batch_max_size
        
        # Start batch processor
        self.batch_processor.start_processing()
        
        # Start server in thread
        self.server_thread = threading.Thread(
            target=self.http_server.serve_forever,
            daemon=True
        )
        self.server_thread.start()
        
        return True
    
    def stop(self):
        """Stop the HTTP server"""
        if self.http_server:
            self.batch_processor.stop_processing()
            self.http_server.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            self.http_server = None
            self.server_thread = None
        
    def is_running(self):
        """Check if server is running"""
        return self.http_server is not None
    
    def get_url(self):
        """Get server URL"""
        return f"http://{self.config.host}:{self.config.port}"