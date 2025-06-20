import os
import time
import gc
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BatchRequest:
    request_id: str
    messages: List[Dict[str, str]]
    settings: Dict[str, Any]
    timestamp: float

class ModelLoader:
    def __init__(self, model_path: str, gui_log_fn, **kwargs):
        self.model_path = model_path
        self.log = gui_log_fn
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.llm = None
        self.is_loaded = False

    def load(self):
        raise NotImplementedError

    def create_completion(self, request_body: Dict, settings: Dict) -> str:
        raise NotImplementedError
    
    def create_batch_completion(self, batch_requests: List[BatchRequest]) -> List[Dict]:
        """Process multiple requests in a batch"""
        results = []
        
        for req in batch_requests:
            try:
                start_time = time.time()
                content = self.create_completion(
                    {"messages": req.messages}, 
                    req.settings
                )
                processing_time = time.time() - start_time
                
                results.append({
                    "request_id": req.request_id,
                    "content": content,
                    "processing_time": processing_time,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "request_id": req.request_id,
                    "content": None,
                    "processing_time": 0,
                    "error": str(e)
                })
        
        return results
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate)"""
        try:
            if hasattr(self, 'tokenizer') and self.tokenizer:
                return len(self.tokenizer.encode(text))
            elif hasattr(self, 'llm') and self.llm:
                return len(self.llm.tokenize(text.encode()))
            else:
                # Rough approximation: 1 token ≈ 4 characters
                return len(text) // 4
        except:
            return len(text) // 4

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        info = {"system_ram_mb": 0, "gpu_memory_gb": []}
        
        try:
            import psutil
            process = psutil.Process()
            info["system_ram_mb"] = process.memory_info().rss / 1024 / 1024
        except:
            pass
        
        # Get GPU memory usage
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Try to get actual GPU memory usage via nvidia-ml-py or nvidia-smi
                    gpu_info = self._get_gpu_memory_usage(i)
                    if gpu_info:
                        info["gpu_memory_gb"].append(gpu_info)
                    else:
                        # Fallback to torch monitoring (may not show llama.cpp usage)
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        info["gpu_memory_gb"].append({
                            "device": i,
                            "allocated": allocated,
                            "reserved": reserved,
                            "total": total,
                            "name": torch.cuda.get_device_name(i)
                        })
        except:
            pass
        
        return info
    
    def _get_gpu_memory_usage(self, device_id: int) -> Optional[Dict]:
        """Get GPU memory usage via nvidia-smi (works for all CUDA applications)"""
        try:
            # Try nvidia-ml-py first (more accurate)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                props = pynvml.nvmlDeviceGetName(handle)
                
                # Handle both bytes and string types
                if isinstance(props, bytes):
                    props = props.decode('utf-8')
                
                total_gb = mem_info.total / 1024**3
                used_gb = mem_info.used / 1024**3
                free_gb = mem_info.free / 1024**3
                
                return {
                    "device": device_id,
                    "allocated": used_gb,
                    "reserved": used_gb,  # For consistency with torch
                    "total": total_gb,
                    "free": free_gb,
                    "name": props
                }
            except ImportError:
                pass
            
            # Fallback to nvidia-smi command
            cmd = [
                "nvidia-smi", 
                "--query-gpu=memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
                f"--id={device_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                line = result.stdout.strip()
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    used_mb = float(parts[0])
                    total_mb = float(parts[1])
                    name = parts[2]
                    
                    used_gb = used_mb / 1024
                    total_gb = total_mb / 1024
                    free_gb = total_gb - used_gb
                    
                    return {
                        "device": device_id,
                        "allocated": used_gb,
                        "reserved": used_gb,
                        "total": total_gb,
                        "free": free_gb,
                        "name": name
                    }
        except Exception as e:
            self.log(f"Error getting GPU {device_id} memory via nvidia-smi: {e}")
        
        return None

    def cleanup(self):
        """Clean up model resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            if hasattr(self, 'llm') and self.llm is not None:
                del self.llm
                self.llm = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
                
            self.is_loaded = False
            self.log("Model resources cleaned up")
        except Exception as e:
            self.log(f"Error during cleanup: {e}")

    def _build_prompt(self, messages: List[Dict], settings: Dict) -> List[Dict]:
        """Build prompt with system and pre-prompts"""
        full_messages = []
        if settings.get('system_prompt'):
            full_messages.append({"role": "system", "content": settings['system_prompt']})
        
        if settings.get('use_pre_prompt') and settings.get('pre_prompt'):
            last_user_message = messages[-1]
            new_content = f"{settings['pre_prompt']}\n\n{last_user_message['content']}"
            full_messages.extend(messages[:-1])
            full_messages.append({"role": "user", "content": new_content})
        else:
            full_messages.extend(messages)
            
        return full_messages

class TransformersLoader(ModelLoader):
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        device_map = self.kwargs.get("device_map", "auto")
        self.log(f"Loading Hugging Face model: {self.model_path} with device_map: '{device_map}'")
        
        # Check available memory before loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free_mem = total_mem - allocated
                self.log(f"GPU {i} ({props.name}): {free_mem:.1f}GB free of {total_mem:.1f}GB total")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                device_map=device_map, 
                load_in_4bit=True,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            device_for_pipeline = -1 if device_map == "auto" else (int(device_map.split(':')[-1]) if 'cuda' in device_map else -1)
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=device_for_pipeline)
            
            self.is_loaded = True
            self.log("Hugging Face model loaded successfully.")
            
            # Log memory usage after loading
            memory_info = self.get_memory_info()
            self.log(f"System RAM usage: {memory_info['system_ram_mb']:.1f}MB")
            for gpu_info in memory_info['gpu_memory_gb']:
                self.log(f"GPU {gpu_info['device']} usage: {gpu_info['allocated']:.1f}GB allocated, {gpu_info['reserved']:.1f}GB reserved")
                
        except Exception as e:
            self.log(f"Error loading Transformers model: {e}")
            raise

    def create_completion(self, request_body: Dict, settings: Dict) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
            
        messages = self._build_prompt(request_body['messages'], settings)
        
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [self.pipeline.tokenizer.eos_token_id, self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        outputs = self.pipeline(
            prompt, 
            max_new_tokens=settings.get('max_tokens', 1024), 
            eos_token_id=terminators, 
            do_sample=True, 
            temperature=settings.get('temperature', 0.7), 
            top_p=settings.get('top_p', 0.9),
            top_k=settings.get('top_k', 40)
        )
        return outputs[0]['generated_text'][len(prompt):].strip()

class GGUFLoader(ModelLoader):
    def _find_maximum_gpu_layers_precisely(self, user_gpu_percentage: float = 100.0) -> tuple:
        """Find maximum GPU layers through precise linear countdown"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return 0, None
            
            total_layers = 80
            
            # Calculate starting point based on theoretical maximum
            starting_layers = self._calculate_starting_point(user_gpu_percentage)
            
            if starting_layers <= 0:
                self.log("No GPU memory available - using CPU only")
                return 0, None
            
            self.log(f"🎯 Starting precise layer fitting")
            self.log(f"   Starting point: {starting_layers} layers")
            self.log(f"   User requested: {user_gpu_percentage:.1f}% GPU usage")
            self.log(f"   Testing layers: {starting_layers} → 1 (until success)")
            
            # Linear countdown: try each layer count until one works
            for test_layers in range(starting_layers, 0, -1):
                # Fixed: Don't use end="" parameter with self.log()
                self.log(f"   Trying {test_layers} layers...")
                
                if self._test_layer_configuration(test_layers, user_gpu_percentage):
                    tensor_split = self._calculate_tensor_split_for_layers(test_layers, user_gpu_percentage)
                    self.log("   ✅ SUCCESS!")
                    self.log(f"🎉 Found precise maximum: {test_layers} layers fit perfectly")
                    return test_layers, tensor_split
                else:
                    self.log("   ❌ Failed (OOM)")
            
            self.log("❌ No GPU layers possible - using CPU only")
            return 0, None
                
        except Exception as e:
            self.log(f"Error in precise search: {e}")
            return 0, None

    def _calculate_starting_point(self, user_gpu_percentage: float) -> int:
        """Calculate smart starting point for linear countdown"""
        try:
            import torch
            
            total_usable_memory = 0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                
                # Minimal buffer for starting point calculation
                buffer = 1.0
                usable_mem = (total_mem - allocated - buffer) * (user_gpu_percentage / 100.0)
                
                if usable_mem > 0:
                    total_usable_memory += usable_mem
                    self.log(f"GPU {i}: {usable_mem:.1f}GB usable for layer fitting")
            
            if total_usable_memory <= 0:
                return 0
            
            # Optimistic starting point: use your empirical data
            # 30 layers = ~17GB, so ~0.57GB per layer
            # Add 10% safety margin: 0.63GB per layer
            optimistic_per_layer = 0.63
            starting_point = int(total_usable_memory / optimistic_per_layer)
            
            # Cap at reasonable maximum (not more than total layers)
            max_reasonable = min(starting_point, 80)
            
            self.log(f"Calculated starting point: {max_reasonable} layers ({total_usable_memory:.1f}GB ÷ {optimistic_per_layer:.2f}GB/layer)")
            
            return max_reasonable
            
        except Exception as e:
            self.log(f"Error calculating starting point: {e}")
            return 40  # Safe fallback

    def _test_layer_configuration(self, layers: int, percentage: float) -> bool:
        """Test if a specific layer configuration can load successfully"""
        try:
            # Build test parameters
            tensor_split = self._calculate_tensor_split_for_layers(layers, percentage)
            
            test_params = {
                'model_path': self.model_path,
                'n_ctx': 512,  # Minimal context for testing
                'n_gpu_layers': layers,
                'vocab_only': True,  # Only load vocabulary for fast testing
                'verbose': False,  # Quiet during testing
                'use_mmap': True,
                'use_mlock': False,
            }
            
            if layers > 0:
                test_params.update({
                    'split_mode': 1,
                    'main_gpu': 0,
                })
                
                if tensor_split and len(tensor_split) > 1:
                    test_params['tensor_split'] = tensor_split
            
            # Try to create the model (vocab_only is much faster)
            from llama_cpp import Llama
            test_llm = Llama(**test_params)
            
            # Clean up immediately
            del test_llm
            
            # Force cleanup
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["out of memory", "cudamalloc failed", "memory", "allocation"]):
                return False  # Memory issue - this config won't work
            else:
                # Log unexpected errors but still return False to be safe
                self.log(f"      Unexpected error during test: {e}")
                return False

    def _calculate_tensor_split_for_layers(self, layers: int, percentage: float) -> list:
        """Calculate tensor split for a specific number of layers"""
        try:
            import torch
            
            if not torch.cuda.is_available() or layers <= 0:
                return None
            
            gpu_count = torch.cuda.device_count()
            if gpu_count <= 1:
                return None
            
            # Get GPU memory info
            gpu_memory = []
            total_memory = 0
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                
                # Use same buffer as main calculation
                buffer = 1.0
                usable_mem = (total_mem - allocated - buffer) * (percentage / 100.0)
                
                if usable_mem > 0:
                    gpu_memory.append(usable_mem)
                    total_memory += usable_mem
                else:
                    gpu_memory.append(0)
            
            if total_memory <= 0:
                return None
            
            # Split proportionally by available memory
            tensor_split = [mem / total_memory for mem in gpu_memory]
            
            return tensor_split
            
        except Exception:
            return None

    def _check_system_resources(self):
        """Check system resources and find maximum GPU layers precisely"""
        try:
            import psutil
            
            # Get file size and RAM info
            file_size_gb = os.path.getsize(self.model_path) / (1024**3)
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            
            self.log(f"Model file size: {file_size_gb:.1f}GB")
            self.log(f"Available system RAM: {available_ram_gb:.1f}GB")
            
            # Get user's desired GPU percentage
            total_layers = 80
            requested_gpu_layers = self.kwargs.get('n_gpu_layers', 0)
            
            if requested_gpu_layers == -1:
                user_gpu_percentage = 100.0
            else:
                user_gpu_percentage = (requested_gpu_layers / total_layers) * 100.0
            
            self.log(f"User requested {user_gpu_percentage:.1f}% GPU usage")
            
            # Use precise countdown to find maximum working layers
            max_gpu_layers, tensor_split = self._find_maximum_gpu_layers_precisely(user_gpu_percentage)
            
            if max_gpu_layers > 0:
                self.kwargs['n_gpu_layers'] = max_gpu_layers
                if tensor_split:
                    self.kwargs['tensor_split'] = tensor_split
                self.log(f"Final configuration: {max_gpu_layers} layers on GPU, {80 - max_gpu_layers} layers on CPU")
            else:
                self.log("Using CPU-only configuration")
                self.kwargs['n_gpu_layers'] = 0
                
        except Exception as e:
            self.log(f"Error in precise resource checking: {e}")

    def load(self):
        from llama_cpp import Llama
        
        self.log(f"Loading GGUF model: {self.model_path}")
        self.log(f"Initial parameters: {self.kwargs}")
        
        # Check system resources and optimize parameters
        self._check_system_resources()
        
        try:
            # Enhanced parameters for better GPU utilization
            load_params = {
                'model_path': self.model_path,
                'n_ctx': self.kwargs.get("n_ctx", 4096),
                'n_gpu_layers': self.kwargs.get("n_gpu_layers", 0),
                'verbose': True,  # Enable verbose logging to see GPU usage
                'use_mmap': True,  # Memory-map the model file
                'use_mlock': False,  # Don't lock memory pages
                'n_threads': None,  # Let llama.cpp decide
                'n_batch': 512,  # Batch size for prompt processing
                'rope_scaling_type': -1,  # Use default RoPE scaling
                'rope_freq_base': 0.0,  # Use model default
                'rope_freq_scale': 0.0,  # Use model default
            }
            
            # Additional GPU-specific parameters
            if self.kwargs.get("n_gpu_layers", 0) > 0:
                load_params.update({
                    'split_mode': 1,  # Split layers across GPUs if multiple GPUs
                    'main_gpu': 0,  # Primary GPU
                })
                
                # Add tensor split if calculated
                tensor_split = self.kwargs.get('tensor_split')
                if tensor_split:
                    load_params['tensor_split'] = tensor_split
            
            self.log(f"Final loading parameters: {load_params}")
            self.log("Initializing llama.cpp model...")
            
            self.llm = Llama(**load_params)
            
            self.is_loaded = True
            self.log("GGUF model loaded successfully.")
            
            # Log actual memory usage
            memory_info = self.get_memory_info()
            self.log(f"Post-load system RAM: {memory_info['system_ram_mb']:.1f}MB")
            for gpu_info in memory_info['gpu_memory_gb']:
                self.log(f"GPU {gpu_info['device']} usage: {gpu_info['allocated']:.1f}GB / {gpu_info['total']:.1f}GB")
                
        except Exception as e:
            self.log(f"Error loading GGUF model: {e}")
            raise e

    def create_completion(self, request_body: Dict, settings: Dict) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
            
        messages = self._build_prompt(request_body['messages'], settings)
        completion = self.llm.create_chat_completion(
            messages=messages,
            temperature=settings.get('temperature', 0.7),
            top_p=settings.get('top_p', 0.9),
            top_k=settings.get('top_k', 40),
            repeat_penalty=settings.get('repeat_penalty', 1.1),
            max_tokens=settings.get('max_tokens', 1024),
        )
        return completion['choices'][0]['message']['content']