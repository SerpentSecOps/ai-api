import base64
import secrets
import hashlib
from typing import Optional, Tuple
from functools import wraps

class AuthManager:
    def __init__(self, username: str = "admin", password: str = "password"):
        self.username = username
        self.password_hash = self._hash_password(password)
        self.sessions = {}  # Simple session storage
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = b'local_llm_salt'  # In production, use random salt per user
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex()
    
    def verify_credentials(self, username: str, password: str) -> bool:
        """Verify username and password"""
        return (username == self.username and 
                self._hash_password(password) == self.password_hash)
    
    def parse_auth_header(self, auth_header: str) -> Optional[Tuple[str, str]]:
        """Parse Basic Auth header"""
        if not auth_header or not auth_header.startswith('Basic '):
            return None
        
        try:
            encoded_credentials = auth_header[6:]  # Remove 'Basic '
            credentials = base64.b64decode(encoded_credentials).decode('utf-8')
            username, password = credentials.split(':', 1)
            return username, password
        except (ValueError, UnicodeDecodeError):
            return None
    
    def authenticate_request(self, headers: dict) -> bool:
        """Authenticate incoming request"""
        auth_header = headers.get('Authorization')
        if not auth_header:
            return False
        
        credentials = self.parse_auth_header(auth_header)
        if not credentials:
            return False
        
        username, password = credentials
        return self.verify_credentials(username, password)
    
    def generate_session_token(self) -> str:
        """Generate a session token"""
        return secrets.token_urlsafe(32)

def require_auth(auth_manager: AuthManager):
    """Decorator for request handlers that require authentication"""
    def decorator(handler_method):
        @wraps(handler_method)
        def wrapper(self, *args, **kwargs):
            if not auth_manager.authenticate_request(dict(self.headers)):
                self.send_response(401)
                self.send_header('WWW-Authenticate', 'Basic realm="LLM Server"')
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {'error': 'Authentication required'}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
                return
            return handler_method(self, *args, **kwargs)
        return wrapper
    return decorator