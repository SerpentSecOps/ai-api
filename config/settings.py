import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import configparser

@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 11434
    cors_origins: list = None
    auth_enabled: bool = True
    auth_username: str = "admin"
    auth_password: str = "password"
    batch_max_size: int = 10
    batch_timeout: float = 30.0
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

@dataclass
class ModelConfig:
    model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    device_map: str = "auto"
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    system_prompt: str = "You are a helpful AI assistant."
    use_pre_prompt: bool = False
    pre_prompt: str = ""

class ConfigManager:
    def __init__(self, config_file: str = "llm_config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.server_config = ServerConfig()
        self.model_config = ModelConfig()
        self.load_config()
    
    def load_config(self):
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            self._load_server_config()
            self._load_model_config()
    
    def _load_server_config(self):
        if 'SERVER' in self.config:
            section = self.config['SERVER']
            self.server_config.host = section.get('host', self.server_config.host)
            self.server_config.port = section.getint('port', self.server_config.port)
            self.server_config.auth_enabled = section.getboolean('auth_enabled', self.server_config.auth_enabled)
            self.server_config.auth_username = section.get('auth_username', self.server_config.auth_username)
            self.server_config.auth_password = section.get('auth_password', self.server_config.auth_password)
            self.server_config.batch_max_size = section.getint('batch_max_size', self.server_config.batch_max_size)
            
            cors_origins = section.get('cors_origins', '*')
            self.server_config.cors_origins = [origin.strip() for origin in cors_origins.split(',')]
    
    def _load_model_config(self):
        if 'MODEL' in self.config:
            section = self.config['MODEL']
            self.model_config.model_path = section.get('model_path', self.model_config.model_path)
            self.model_config.n_ctx = section.getint('n_ctx', self.model_config.n_ctx)
            self.model_config.n_gpu_layers = section.getint('n_gpu_layers', self.model_config.n_gpu_layers)
            self.model_config.max_tokens = section.getint('max_tokens', self.model_config.max_tokens)
            self.model_config.temperature = section.getfloat('temperature', self.model_config.temperature)
            self.model_config.top_p = section.getfloat('top_p', self.model_config.top_p)
            self.model_config.top_k = section.getint('top_k', self.model_config.top_k)
            self.model_config.repeat_penalty = section.getfloat('repeat_penalty', self.model_config.repeat_penalty)
            self.model_config.system_prompt = section.get('system_prompt', self.model_config.system_prompt)
            self.model_config.use_pre_prompt = section.getboolean('use_pre_prompt', self.model_config.use_pre_prompt)
            self.model_config.pre_prompt = section.get('pre_prompt', self.model_config.pre_prompt)
    
    def save_config(self):
        # Save server config
        if 'SERVER' not in self.config:
            self.config['SERVER'] = {}
        
        server_section = self.config['SERVER']
        server_section['host'] = self.server_config.host
        server_section['port'] = str(self.server_config.port)
        server_section['auth_enabled'] = str(self.server_config.auth_enabled)
        server_section['auth_username'] = self.server_config.auth_username
        server_section['auth_password'] = self.server_config.auth_password
        server_section['batch_max_size'] = str(self.server_config.batch_max_size)
        server_section['cors_origins'] = ','.join(self.server_config.cors_origins)
        
        # Save model config
        if 'MODEL' not in self.config:
            self.config['MODEL'] = {}
        
        model_section = self.config['MODEL']
        model_section['model_path'] = self.model_config.model_path
        model_section['n_ctx'] = str(self.model_config.n_ctx)
        model_section['n_gpu_layers'] = str(self.model_config.n_gpu_layers)
        model_section['max_tokens'] = str(self.model_config.max_tokens)
        model_section['temperature'] = str(self.model_config.temperature)
        model_section['top_p'] = str(self.model_config.top_p)
        model_section['top_k'] = str(self.model_config.top_k)
        model_section['repeat_penalty'] = str(self.model_config.repeat_penalty)
        model_section['system_prompt'] = self.model_config.system_prompt
        model_section['use_pre_prompt'] = str(self.model_config.use_pre_prompt)
        model_section['pre_prompt'] = self.model_config.pre_prompt
        
        with open(self.config_file, 'w') as f:
            self.config.write(f)