import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
import threading
import queue
import time
import os
import sys
import struct
from typing import Optional

class ControlPanelGUI(tk.Tk):
    def __init__(self, app_state):
        super().__init__()
        self.app_state = app_state
        
        # Import torch here to avoid issues if not available
        try:
            import torch
            self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            self.gpu_count = 0
        
        self.title("Local LLM Control Panel (Enhanced)")
        self.geometry("900x950")
        
        self.log_queue = queue.Queue()
        self.memory_monitor_active = False
        
        self._init_vars()
        self._load_saved_settings()
        self._create_widgets()
        
        # Start GUI update loops
        self.process_log_queue()
        self.start_memory_monitoring()

    def _init_vars(self):
        """Initialize GUI variables"""
        # Model settings
        self.model_path_var = tk.StringVar(value="meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.context_size_var = tk.StringVar(value="4096")
        self.total_layers_var = tk.StringVar(value="80")
        self.gpu_offload_percent_var = tk.DoubleVar(value=100.0)
        
        # Server settings
        self.port_var = tk.StringVar(value="11434")
        self.host_var = tk.StringVar(value="localhost")
        self.auth_enabled_var = tk.BooleanVar(value=True)
        self.auth_username_var = tk.StringVar(value="admin")
        self.auth_password_var = tk.StringVar(value="password")
        self.cors_origins_var = tk.StringVar(value="*")
        self.batch_size_var = tk.StringVar(value="10")
        
        # Generation parameters
        self.max_tokens_var = tk.StringVar(value="1024")
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.top_p_var = tk.DoubleVar(value=0.9)
        self.top_k_var = tk.IntVar(value=40)
        self.repeat_penalty_var = tk.DoubleVar(value=1.1)
        
        # Prompt settings
        self.use_pre_prompt_var = tk.BooleanVar(value=False)

    def _load_saved_settings(self):
        """Load settings from config manager"""
        config = self.app_state.config_manager
        
        # Server settings
        self.host_var.set(config.server_config.host)
        self.port_var.set(str(config.server_config.port))
        self.auth_enabled_var.set(config.server_config.auth_enabled)
        self.auth_username_var.set(config.server_config.auth_username)
        self.auth_password_var.set(config.server_config.auth_password)
        self.cors_origins_var.set(','.join(config.server_config.cors_origins))
        self.batch_size_var.set(str(config.server_config.batch_max_size))
        
        # Model settings
        self.model_path_var.set(config.model_config.model_path)
        self.context_size_var.set(str(config.model_config.n_ctx))
        self.max_tokens_var.set(str(config.model_config.max_tokens))
        self.temperature_var.set(config.model_config.temperature)
        self.top_p_var.set(config.model_config.top_p)
        self.top_k_var.set(config.model_config.top_k)
        self.repeat_penalty_var.set(config.model_config.repeat_penalty)
        self.use_pre_prompt_var.set(config.model_config.use_pre_prompt)

    def _create_widgets(self):
        """Create all GUI widgets"""
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create tabs
        model_tab = ttk.Frame(notebook)
        server_tab = ttk.Frame(notebook)
        params_tab = ttk.Frame(notebook)
        prompt_tab = ttk.Frame(notebook)
        metrics_tab = ttk.Frame(notebook)
        
        notebook.add(model_tab, text='Model')
        notebook.add(server_tab, text='Server')
        notebook.add(params_tab, text='Parameters')
        notebook.add(prompt_tab, text='Prompting')
        notebook.add(metrics_tab, text='Metrics')
        
        # Create tab content
        self._create_model_widgets(model_tab)
        self._create_server_widgets(server_tab)
        self._create_params_widgets(params_tab)
        self._create_prompt_widgets(prompt_tab)
        self._create_metrics_widgets(metrics_tab)
        
        # Create bottom panels
        self._create_control_widgets(main_frame)
        self._create_log_widgets(main_frame)

    def _create_model_widgets(self, parent):
        """Create model configuration widgets"""
        frame = tk.LabelFrame(parent, text="Model Configuration", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Model path
        tk.Label(frame, text="Model Path or HF Repo:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_entry = tk.Entry(frame, textvariable=self.model_path_var, width=50)
        self.model_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        tk.Button(frame, text="Browse GGUF...", command=self.browse_for_model).grid(row=0, column=3, padx=5)
        
        # Context size
        tk.Label(frame, text="Context Size (n_ctx):").grid(row=1, column=0, sticky=tk.W, pady=2)
        tk.Entry(frame, textvariable=self.context_size_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # GPU settings frame
        gpu_frame = tk.LabelFrame(frame, text="GPU Offload (GGUF)", padx=5, pady=5)
        gpu_frame.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=10)
        
        tk.Label(gpu_frame, text="Total Layers:").grid(row=0, column=0, sticky=tk.W, padx=5)
        layer_frame = tk.Frame(gpu_frame)
        layer_frame.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        self.total_layers_entry = tk.Entry(layer_frame, textvariable=self.total_layers_var, width=10)
        self.total_layers_entry.pack(side=tk.LEFT)
        tk.Button(layer_frame, text="Auto-Detect", command=self.detect_gguf_layers).pack(side=tk.LEFT, padx=5)
        
        tk.Label(gpu_frame, text="GPU Offload %:").grid(row=1, column=0, sticky=tk.W, padx=5)
        slider_frame = tk.Frame(gpu_frame)
        slider_frame.grid(row=1, column=1, sticky=tk.EW, padx=5)
        
        offload_slider = tk.Scale(slider_frame, from_=0, to=100, resolution=1, 
                                orient=tk.HORIZONTAL, variable=self.gpu_offload_percent_var)
        offload_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(slider_frame, textvariable=self.gpu_offload_percent_var, width=6).pack(side=tk.LEFT)
        
        # Configure grid weights
        frame.grid_columnconfigure(1, weight=1)
        gpu_frame.grid_columnconfigure(1, weight=1)

    def _create_server_widgets(self, parent):
        """Create server configuration widgets"""
        frame = tk.LabelFrame(parent, text="Server Configuration", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Basic server settings
        basic_frame = tk.LabelFrame(frame, text="Connection", padx=5, pady=5)
        basic_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(basic_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, padx=5)
        tk.Entry(basic_frame, textvariable=self.host_var, width=15).grid(row=0, column=1, padx=5)
        
        tk.Label(basic_frame, text="Port:").grid(row=0, column=2, sticky=tk.W, padx=5)
        tk.Entry(basic_frame, textvariable=self.port_var, width=10).grid(row=0, column=3, padx=5)
        
        tk.Label(basic_frame, text="CORS Origins:").grid(row=1, column=0, sticky=tk.W, padx=5)
        tk.Entry(basic_frame, textvariable=self.cors_origins_var, width=30).grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=5)
        
        tk.Label(basic_frame, text="Max Batch Size:").grid(row=2, column=0, sticky=tk.W, padx=5)
        tk.Entry(basic_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, padx=5)
        
        basic_frame.grid_columnconfigure(1, weight=1)
        
        # Authentication settings
        auth_frame = tk.LabelFrame(frame, text="Authentication", padx=5, pady=5)
        auth_frame.pack(fill=tk.X, pady=5)
        
        tk.Checkbutton(auth_frame, text="Enable Authentication", 
                      variable=self.auth_enabled_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        tk.Label(auth_frame, text="Username:").grid(row=1, column=0, sticky=tk.W, padx=5)
        tk.Entry(auth_frame, textvariable=self.auth_username_var, width=20).grid(row=1, column=1, padx=5, sticky=tk.W)
        
        tk.Label(auth_frame, text="Password:").grid(row=2, column=0, sticky=tk.W, padx=5)
        tk.Entry(auth_frame, textvariable=self.auth_password_var, width=20, show="*").grid(row=2, column=1, padx=5, sticky=tk.W)

    def _create_params_widgets(self, parent):
        """Create generation parameter widgets"""
        frame = tk.LabelFrame(parent, text="Generation Parameters", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Max tokens
        tk.Label(frame, text="Max New Tokens:").grid(row=0, column=0, sticky=tk.W, pady=4)
        tk.Entry(frame, textvariable=self.max_tokens_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Temperature
        tk.Label(frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=4)
        temp_frame = tk.Frame(frame)
        temp_frame.grid(row=1, column=1, sticky=tk.EW, padx=5)
        tk.Scale(temp_frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, 
                variable=self.temperature_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(temp_frame, textvariable=self.temperature_var, width=6).pack(side=tk.LEFT)
        
        # Top P
        tk.Label(frame, text="Top P:").grid(row=2, column=0, sticky=tk.W, pady=4)
        top_p_frame = tk.Frame(frame)
        top_p_frame.grid(row=2, column=1, sticky=tk.EW, padx=5)
        tk.Scale(top_p_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, 
                variable=self.top_p_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(top_p_frame, textvariable=self.top_p_var, width=6).pack(side=tk.LEFT)
        
        # Top K
        tk.Label(frame, text="Top K:").grid(row=3, column=0, sticky=tk.W, pady=4)
        top_k_frame = tk.Frame(frame)
        top_k_frame.grid(row=3, column=1, sticky=tk.EW, padx=5)
        tk.Scale(top_k_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                variable=self.top_k_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(top_k_frame, textvariable=self.top_k_var, width=6).pack(side=tk.LEFT)
        
        # Repeat penalty
        tk.Label(frame, text="Repeat Penalty:").grid(row=4, column=0, sticky=tk.W, pady=4)
        repeat_frame = tk.Frame(frame)
        repeat_frame.grid(row=4, column=1, sticky=tk.EW, padx=5)
        tk.Scale(repeat_frame, from_=1.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, 
                variable=self.repeat_penalty_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(repeat_frame, textvariable=self.repeat_penalty_var, width=6).pack(side=tk.LEFT)
        
        frame.grid_columnconfigure(1, weight=1)

    def _create_prompt_widgets(self, parent):
        """Create prompt engineering widgets"""
        frame = tk.LabelFrame(parent, text="Prompt Engineering", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # System prompt
        sys_frame = tk.LabelFrame(frame, text="System Prompt", padx=5, pady=5)
        sys_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.system_prompt_text = scrolledtext.ScrolledText(sys_frame, height=8, wrap=tk.WORD)
        self.system_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.system_prompt_text.insert(tk.END, "You are a helpful AI assistant.")
        
        # Pre-instruction
        pre_frame = tk.LabelFrame(frame, text="Pre-instruction (Role-play)", padx=5, pady=5)
        pre_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Checkbutton(pre_frame, text="Enable Pre-instruction", 
                      variable=self.use_pre_prompt_var).pack(anchor=tk.W)
        
        self.pre_prompt_text = scrolledtext.ScrolledText(pre_frame, height=4, wrap=tk.WORD)
        self.pre_prompt_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def _create_metrics_widgets(self, parent):
        """Create metrics display widgets"""
        frame = tk.LabelFrame(parent, text="Performance Metrics", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Stats display
        stats_frame = tk.LabelFrame(frame, text="Statistics", padx=5, pady=5)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, wrap=tk.WORD, state='disabled',
                                 bg="#f0f0f0", font=("Consolas", 9))
        stats_scrollbar = tk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="Refresh Metrics", command=self.refresh_metrics).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Export Metrics", command=self.export_metrics).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear Metrics", command=self.clear_metrics).pack(side=tk.LEFT, padx=5)

    def _create_control_widgets(self, parent):
        """Create server control widgets"""
        control_frame = tk.LabelFrame(parent, text="Server Control", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Status display
        self.status_label = tk.Label(control_frame, text="Server is stopped.", 
                                   font=("Helvetica", 10, "italic"))
        self.status_label.pack(pady=5)
        
        # Memory usage display
        self.memory_label = tk.Label(control_frame, text="Memory: N/A", 
                                   font=("Helvetica", 9))
        self.memory_label.pack()
        
        # Control buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(pady=10)
        
        self.start_button = tk.Button(btn_frame, text="Start Server", 
                                    command=self.start_server_flow, 
                                    bg="#90EE90", font=("Helvetica", 10, "bold"))
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(btn_frame, text="Stop Server", 
                                   command=self.stop_server_flow, 
                                   state=tk.DISABLED, 
                                   bg="#F08080", font=("Helvetica", 10, "bold"))
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="System Info", command=self.show_system_info).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5)

    def _create_log_widgets(self, parent):
        """Create log display widgets"""
        log_frame = tk.LabelFrame(parent, text="Live Log", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        log_container = tk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_container, wrap=tk.WORD, 
                                                state='disabled', 
                                                bg="#2E2E2E", fg="#D3D3D3",
                                                font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log control buttons
        log_btn_frame = tk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(log_btn_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        tk.Button(log_btn_frame, text="Export Log", command=self.export_log).pack(side=tk.LEFT, padx=5)

    # Event handlers and utility methods
    def log_message(self, msg):
        """Add message to log queue"""
        self.log_queue.put(msg)

    def process_log_queue(self):
        """Process messages from log queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state='normal')
                self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
                self.log_text.config(state='disabled')
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        
        self.after(100, self.process_log_queue)

    def browse_for_model(self):
        """Browse for GGUF model file"""
        filepath = filedialog.askopenfilename(
            title="Select GGUF Model",
            filetypes=[("GGUF Models", "*.gguf"), ("All files", "*.*")]
        )
        if filepath:
            self.model_path_var.set(filepath)
            self.detect_gguf_layers()

    def detect_gguf_layers(self):
        """Auto-detect layers in GGUF model - updated for llama-cpp-python 0.3.x+"""
        model_path = self.model_path_var.get()
        if not model_path.lower().endswith(".gguf") or not os.path.exists(model_path):
            messagebox.showinfo("Info", "Layer detection only works for local GGUF files.")
            return
        
        self.log_message(f"Detecting layers in {os.path.basename(model_path)}...")
        
        # Method 1: Try parsing GGUF file directly
        layer_count = self._detect_layers_gguf_parser(model_path)
        
        # Method 2: Try loading model temporarily (current API)
        if layer_count is None:
            layer_count = self._detect_layers_model_inspection(model_path)
        
        # Method 3: Use common defaults based on model name
        if layer_count is None:
            layer_count = self._detect_layers_by_name(model_path)
        
        if layer_count:
            self.total_layers_var.set(str(layer_count))
            self.log_message(f"Detected {layer_count} layers.")
        else:
            self.log_message("Could not automatically detect layer count.")
            messagebox.showwarning("Warning", 
                "Could not detect layer count automatically.\n"
                "Please enter the layer count manually or use these common values:\n\n"
                "• 7B models: ~32 layers\n"
                "• 8B models: ~32 layers\n"
                "• 13B models: ~40 layers\n"
                "• 30B models: ~60 layers\n"
                "• 70B models: ~80 layers\n"
                "• 405B models: ~126 layers"
            )

    def _detect_layers_gguf_parser(self, model_path: str) -> Optional[int]:
        """Method 1: Parse GGUF file directly to extract layer count"""
        try:
            self.log_message("Parsing GGUF file headers...")
            
            with open(model_path, 'rb') as f:
                # Read GGUF magic number
                magic = f.read(4)
                if magic != b'GGUF':
                    self.log_message("Not a valid GGUF file")
                    return None
                
                # Read version
                version = struct.unpack('<I', f.read(4))[0]
                self.log_message(f"GGUF version: {version}")
                
                if version < 1 or version > 3:
                    self.log_message(f"Unsupported GGUF version: {version}")
                    return None
                
                # Read tensor and metadata counts
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                self.log_message(f"Found {metadata_count} metadata entries")
                
                # Parse metadata entries
                for i in range(metadata_count):
                    try:
                        # Read key length and key
                        key_len = struct.unpack('<Q', f.read(8))[0]
                        if key_len > 1000:  # Sanity check
                            break
                        
                        key = f.read(key_len).decode('utf-8', errors='ignore')
                        
                        # Read value type
                        value_type = struct.unpack('<I', f.read(4))[0]
                        
                        # Parse value based on type
                        if value_type == 4:  # UINT32
                            value = struct.unpack('<I', f.read(4))[0]
                        elif value_type == 5:  # INT32
                            value = struct.unpack('<i', f.read(4))[0]
                        elif value_type == 6:  # FLOAT32
                            value = struct.unpack('<f', f.read(4))[0]
                        elif value_type == 7:  # BOOL
                            value = struct.unpack('<?', f.read(1))[0]
                        elif value_type == 8:  # STRING
                            str_len = struct.unpack('<Q', f.read(8))[0]
                            if str_len > 10000:  # Sanity check
                                break
                            value = f.read(str_len).decode('utf-8', errors='ignore')
                        elif value_type == 9:  # ARRAY - skip for now
                            array_type = struct.unpack('<I', f.read(4))[0]
                            array_len = struct.unpack('<Q', f.read(8))[0]
                            # Skip array contents
                            if array_type in [4, 5]:  # INT32/UINT32
                                f.read(4 * array_len)
                            elif array_type == 6:  # FLOAT32
                                f.read(4 * array_len)
                            elif array_type == 7:  # BOOL
                                f.read(array_len)
                            else:
                                break  # Unknown array type
                            continue
                        else:
                            self.log_message(f"Unknown value type: {value_type}")
                            break
                        
                        # Check for layer count keys
                        if key in ['llama.block_count', 'block_count', 'n_layer', 'n_layers']:
                            if isinstance(value, (int, float)):
                                self.log_message(f"Found {key} = {int(value)}")
                                return int(value)
                                
                    except Exception as e:
                        self.log_message(f"Error parsing metadata entry {i}: {e}")
                        break
                        
        except Exception as e:
            self.log_message(f"GGUF parsing failed: {e}")
        
        return None

    def _detect_layers_model_inspection(self, model_path: str) -> Optional[int]:
        """Method 2: Load model with minimal resources and inspect structure"""
        try:
            from llama_cpp import Llama
            
            self.log_message("Loading model for inspection...")
            
            # Load model with minimal resources
            llm = Llama(
                model_path=model_path,
                n_ctx=256,  # Minimal context
                n_gpu_layers=0,  # CPU only for detection
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                n_threads=1,  # Single thread
                n_batch=1     # Minimal batch
            )
            
            layer_count = None
            
            # Try various ways to get layer count from the loaded model
            try:
                # Method 1: Direct attributes
                if hasattr(llm, 'n_layers'):
                    layer_count = llm.n_layers
                elif hasattr(llm, '_model') and hasattr(llm._model, 'n_layers'):
                    layer_count = llm._model.n_layers
                elif hasattr(llm, 'model') and hasattr(llm.model, 'n_layers'):
                    layer_count = llm.model.n_layers
                
                # Method 2: Try to access internal model structure
                if not layer_count and hasattr(llm, '_model'):
                    model_obj = llm._model
                    for attr_name in dir(model_obj):
                        if 'layer' in attr_name.lower() and ('count' in attr_name.lower() or 'n_' in attr_name.lower()):
                            try:
                                attr_value = getattr(model_obj, attr_name)
                                if isinstance(attr_value, int) and 10 <= attr_value <= 200:
                                    layer_count = attr_value
                                    self.log_message(f"Found layer count via {attr_name}: {layer_count}")
                                    break
                            except:
                                continue
                
                # Method 3: Try to get model parameters and count transformer blocks
                if not layer_count:
                    try:
                        # This is a more advanced approach - try to count actual transformer blocks
                        # by looking at the model's parameter names
                        if hasattr(llm, '_model') and hasattr(llm._model, 'get_tensor'):
                            # Count how many transformer blocks exist by looking for block patterns
                            block_numbers = set()
                            # This would require access to tensor names, which might not be available
                            # in the current API
                    except:
                        pass
                        
            except Exception as e:
                self.log_message(f"Error inspecting model attributes: {e}")
            
            # Clean up
            try:
                del llm
            except:
                pass
            
            if layer_count and isinstance(layer_count, int) and layer_count > 0:
                self.log_message(f"Model inspection found {layer_count} layers")
                return layer_count
                
        except Exception as e:
            self.log_message(f"Model inspection failed: {e}")
        
        return None

    def _detect_layers_by_name(self, model_path: str) -> Optional[int]:
        """Method 3: Estimate layer count based on model name patterns"""
        filename = os.path.basename(model_path).lower()
        
        # Enhanced patterns for different model sizes and families
        patterns = {
            # Llama 3.1/3.2 series
            'llama-3.1-8b': 32,
            'llama-3.1-70b': 80,
            'llama-3.1-405b': 126,
            'llama-3.2-1b': 16,
            'llama-3.2-3b': 28,
            'llama-3.2-11b': 32,
            'llama-3.2-90b': 88,
            
            # Llama 3 series
            'llama-3-8b': 32,
            'llama-3-70b': 80,
            
            # Llama 2 series
            'llama-2-7b': 32,
            'llama-2-13b': 40,
            'llama-2-70b': 80,
            
            # Generic patterns (fallback)
            '1b': 16,
            '3b': 28,
            '7b': 32,
            '8b': 32,
            '11b': 32,
            '13b': 40,
            '30b': 60,
            '33b': 60,
            '70b': 80,
            '65b': 80,
            '90b': 88,
            '180b': 96,
            '405b': 126,
        }
        
        # Try specific patterns first, then generic ones
        for pattern, layers in patterns.items():
            if pattern in filename:
                self.log_message(f"Estimated {layers} layers based on pattern '{pattern}' in filename")
                return layers
        
        # If it's a 70B model (like your file), assume 80 layers
        if '70b' in filename or '70B' in os.path.basename(model_path):
            self.log_message("Detected 70B model, using 80 layers")
            return 80
        
        # Default fallback
        self.log_message("No pattern matched, using default 32 layers")
        return 32

    def set_status(self, text, color="black"):
        """Update status label"""
        self.status_label.config(text=text, fg=color)
        self.update_idletasks()

    def start_memory_monitoring(self):
        """Start memory usage monitoring"""
        self.memory_monitor_active = True
        self._update_memory_display()

    def _update_memory_display(self):
        """Update memory usage display"""
        if not self.memory_monitor_active:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            status_text = f"RAM: {memory_mb:.1f}MB ({memory_percent:.1f}%)"
            
            # Add GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        status_text += f" | GPU{i}: {allocated:.1f}GB"
            except:
                pass
            
            self.memory_label.config(text=status_text)
            
        except Exception:
            self.memory_label.config(text="Memory: N/A")
        
        # Schedule next update
        self.after(5000, self._update_memory_display)

    def _collect_settings(self):
        """Collect all settings from GUI"""
        from config.settings import ServerConfig, ModelConfig
        
        # Update server config
        server_config = self.app_state.config_manager.server_config
        server_config.host = self.host_var.get()
        server_config.port = int(self.port_var.get())
        server_config.auth_enabled = self.auth_enabled_var.get()
        server_config.auth_username = self.auth_username_var.get()
        server_config.auth_password = self.auth_password_var.get()
        server_config.cors_origins = [s.strip() for s in self.cors_origins_var.get().split(',')]
        server_config.batch_max_size = int(self.batch_size_var.get())
        
        # Update model config
        model_config = self.app_state.config_manager.model_config
        model_config.model_path = self.model_path_var.get()
        model_config.n_ctx = int(self.context_size_var.get())
        model_config.max_tokens = int(self.max_tokens_var.get())
        model_config.temperature = self.temperature_var.get()
        model_config.top_p = self.top_p_var.get()
        model_config.top_k = self.top_k_var.get()
        model_config.repeat_penalty = self.repeat_penalty_var.get()
        model_config.system_prompt = self.system_prompt_text.get("1.0", tk.END).strip()
        model_config.use_pre_prompt = self.use_pre_prompt_var.get()
        model_config.pre_prompt = self.pre_prompt_text.get("1.0", tk.END).strip()
        
        # Calculate GPU layers
        total_layers = int(self.total_layers_var.get())
        offload_percent = self.gpu_offload_percent_var.get()
        model_config.n_gpu_layers = int((offload_percent / 100.0) * total_layers)
        
        return server_config, model_config

    def start_server_flow(self):
        """Start the server"""
        try:
            server_config, model_config = self._collect_settings()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {e}")
            return
        
        if not model_config.model_path:
            messagebox.showerror("Error", "Model path cannot be empty.")
            return
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.set_status("Starting...", color="blue")
        
        # Start server in background thread
        threading.Thread(target=self._server_thread_worker, 
                        args=(server_config, model_config), 
                        daemon=True).start()

    def _server_thread_worker(self, server_config, model_config):
        """Server worker thread"""
        try:
            from api.server import LLMServer
            from core.model_loader import TransformersLoader, GGUFLoader
            
            # Load model
            self.set_status(f"Loading model: {os.path.basename(model_config.model_path)}", color="blue")
            
            if model_config.model_path.lower().endswith(".gguf"):
                loader_class = GGUFLoader
                model_kwargs = {
                    'n_gpu_layers': model_config.n_gpu_layers,
                    'n_ctx': model_config.n_ctx
                }
            else:
                loader_class = TransformersLoader
                model_kwargs = {'device_map': model_config.device_map}
            
            self.app_state.model_loader = loader_class(
                model_config.model_path, 
                self.log_message, 
                **model_kwargs
            )
            self.app_state.model_loader.load()
            
            # Create and start server
            self.set_status("Starting HTTP server...", color="blue")
            self.app_state.server = LLMServer(server_config)
            
            # Convert model config to dict for server
            model_settings = {
                'max_tokens': model_config.max_tokens,
                'temperature': model_config.temperature,
                'top_p': model_config.top_p,
                'top_k': model_config.top_k,
                'repeat_penalty': model_config.repeat_penalty,
                'system_prompt': model_config.system_prompt,
                'use_pre_prompt': model_config.use_pre_prompt,
                'pre_prompt': model_config.pre_prompt,
            }
            
            self.app_state.server.set_model_loader(self.app_state.model_loader, model_settings)
            
            if self.app_state.server.start():
                url = self.app_state.server.get_url()
                self.log_message(f"Server started on {url}")
                self.set_status(f"Server running on {url}", color="green")
            else:
                raise Exception("Failed to start server")
                
        except Exception as e:
            self.log_message(f"FATAL ERROR: {e}")
            self.set_status("Error. Check logs.", color="red")
            messagebox.showerror("Server Error", f"Failed to start server: {e}")
            self.stop_server_flow()

    def stop_server_flow(self):
        """Stop the server"""
        self.log_message("Stopping server...")
        self.set_status("Stopping...", color="orange")
        
        # Stop server
        if self.app_state.server:
            self.app_state.server.stop()
            self.app_state.server = None
        
        # Clean up model
        if self.app_state.model_loader:
            self.app_state.model_loader.cleanup()
            self.app_state.model_loader = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.set_status("Server is stopped.", color="black")
        
        # Save settings
        self.save_settings()

    def save_settings(self):
        """Save current settings"""
        try:
            self._collect_settings()  # This updates the config objects
            self.app_state.config_manager.save_config()
            self.log_message("Settings saved successfully.")
            messagebox.showinfo("Settings", "Settings saved successfully.")
        except Exception as e:
            self.log_message(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def refresh_metrics(self):
        """Refresh metrics display"""
        try:
            from core.metrics import metrics_collector
            stats = metrics_collector.get_stats()
            recent = metrics_collector.get_recent_metrics(10)
            
            self.stats_text.config(state='normal')
            self.stats_text.delete('1.0', tk.END)
            
            # Display stats
            self.stats_text.insert(tk.END, "=== PERFORMANCE STATISTICS ===\n\n")
            self.stats_text.insert(tk.END, f"Total Requests: {stats['total_requests']}\n")
            self.stats_text.insert(tk.END, f"Successful: {stats['successful_requests']}\n")
            self.stats_text.insert(tk.END, f"Failed: {stats['failed_requests']}\n")
            self.stats_text.insert(tk.END, f"Success Rate: {stats['success_rate']:.1f}%\n")
            self.stats_text.insert(tk.END, f"Total Tokens: {stats['total_tokens_processed']}\n")
            self.stats_text.insert(tk.END, f"Avg Tokens/sec: {stats['average_tokens_per_second']}\n")
            self.stats_text.insert(tk.END, f"Avg Response Time: {stats['average_response_time']:.3f}s\n")
            self.stats_text.insert(tk.END, f"Current Queue: {stats['current_queue_size']}\n")
            self.stats_text.insert(tk.END, f"Peak Queue: {stats['peak_queue_size']}\n")
            self.stats_text.insert(tk.END, f"Active Requests: {stats['active_requests']}\n")
            
            # Display recent requests
            if recent:
                self.stats_text.insert(tk.END, "\n=== RECENT REQUESTS ===\n\n")
                for req in recent[-5:]:  # Show last 5
                    self.stats_text.insert(tk.END, f"ID: {req['request_id'][:8]}...\n")
                    self.stats_text.insert(tk.END, f"  Time: {req['processing_time']:.3f}s\n")
                    self.stats_text.insert(tk.END, f"  Tokens: {req['input_tokens']} + {req['output_tokens']}\n")
                    self.stats_text.insert(tk.END, f"  Speed: {req['tokens_per_second']:.1f} tok/s\n")
                    if req['error']:
                        self.stats_text.insert(tk.END, f"  Error: {req['error']}\n")
                    self.stats_text.insert(tk.END, "\n")
            
            self.stats_text.config(state='disabled')
            
        except Exception as e:
            self.log_message(f"Error refreshing metrics: {e}")

    def export_metrics(self):
        """Export metrics to file"""
        try:
            from core.metrics import metrics_collector
            filepath = filedialog.asksaveasfilename(
                title="Export Metrics",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filepath:
                metrics_collector.export_metrics(filepath)
                self.log_message(f"Metrics exported to {filepath}")
                messagebox.showinfo("Export", f"Metrics exported to {filepath}")
        except Exception as e:
            self.log_message(f"Error exporting metrics: {e}")
            messagebox.showerror("Error", f"Failed to export metrics: {e}")

    def clear_metrics(self):
        """Clear metrics history"""
        if messagebox.askyesno("Clear Metrics", "Are you sure you want to clear all metrics history?"):
            try:
                from core.metrics import metrics_collector
                metrics_collector.metrics_history.clear()
                metrics_collector.stats = type(metrics_collector.stats)()
                self.refresh_metrics()
                self.log_message("Metrics cleared.")
            except Exception as e:
                self.log_message(f"Error clearing metrics: {e}")

    def clear_log(self):
        """Clear log display"""
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')

    def export_log(self):
        """Export log to file"""
        try:
            filepath = filedialog.asksaveasfilename(
                title="Export Log",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filepath:
                log_content = self.log_text.get('1.0', tk.END)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                self.log_message(f"Log exported to {filepath}")
                messagebox.showinfo("Export", f"Log exported to {filepath}")
        except Exception as e:
            self.log_message(f"Error exporting log: {e}")
            messagebox.showerror("Error", f"Failed to export log: {e}")

    def show_system_info(self):
        """Show system information dialog"""
        try:
            info_window = tk.Toplevel(self)
            info_window.title("System Information")
            info_window.geometry("700x500")
            
            info_text = tk.Text(info_window, wrap=tk.WORD, font=("Consolas", 10))
            info_scrollbar = tk.Scrollbar(info_window, orient="vertical", command=info_text.yview)
            info_text.configure(yscrollcommand=info_scrollbar.set)
            
            info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
            
            # Gather system info
            info = f"Python Executable: {sys.executable}\n\n"
            
            try:
                import torch
                info += f"PyTorch Version: {torch.__version__}\n"
                cuda_available = torch.cuda.is_available()
                info += f"CUDA Available: {cuda_available}\n"
                if cuda_available:
                    info += f"CUDA Version: {torch.version.cuda}\n"
                    count = torch.cuda.device_count()
                    info += f"GPU Count: {count}\n"
                    for i in range(count):
                        info += f"  GPU {i}: {torch.cuda.get_device_name(i)}\n"
            except ImportError:
                info += "PyTorch: Not installed\n"
            
            try:
                import transformers
                info += f"Transformers Version: {transformers.__version__}\n"
            except ImportError:
                info += "Transformers: Not installed\n"
            
            try:
                import llama_cpp
                info += f"llama-cpp-python Version: {llama_cpp.__version__}\n"
            except ImportError:
                info += "llama-cpp-python: Not installed\n"
            
            try:
                import psutil
                memory = psutil.virtual_memory()
                info += f"\nSystem Memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available\n"
                info += f"CPU Count: {psutil.cpu_count()} cores\n"
            except ImportError:
                info += "psutil: Not installed (memory info unavailable)\n"
            
            info_text.insert(tk.END, info)
            info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve system info: {e}")

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit? This will stop the server and save settings."):
            self.memory_monitor_active = False
            self.save_settings()
            self.stop_server_flow()
            time.sleep(1)  # Give cleanup time
            self.destroy()