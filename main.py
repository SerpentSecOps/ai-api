#!/usr/bin/env python3
"""
Local LLM Server - Modular Edition
A desktop application with Tkinter GUI to control a local LLM server.
Supports both Hugging Face Transformers and GGUF models with advanced controls.

Features:
- CORS support for web integration
- Basic authentication
- Batch processing for multiple requests
- Performance metrics and monitoring
- Modular architecture
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import threading
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import ConfigManager, ServerConfig, ModelConfig
from core.model_loader import TransformersLoader, GGUFLoader
from core.auth import AuthManager
from core.metrics import metrics_collector
from api.server import LLMServer
from gui.control_panel import ControlPanelGUI

# Global application state
class AppState:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.server = None
        self.model_loader = None
        self.gui = None
        
    def cleanup(self):
        """Clean up all resources"""
        if self.server:
            self.server.stop()
        if self.model_loader:
            self.model_loader.cleanup()

# Global instance
app_state = AppState()

def main():
    """Main application entry point"""
    try:
        # Create and configure GUI
        app_state.gui = ControlPanelGUI(app_state)
        app_state.gui.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Load configuration
        app_state.config_manager.load_config()
        
        # Start GUI main loop
        app_state.gui.mainloop()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_and_exit()
    except Exception as e:
        print(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"Application encountered a fatal error:\n{e}")
        cleanup_and_exit()

def on_closing():
    """Handle application closing"""
    if messagebox.askokcancel("Quit", "Do you want to quit? This will stop the server and save settings."):
        cleanup_and_exit()

def cleanup_and_exit():
    """Clean up resources and exit"""
    try:
        app_state.config_manager.save_config()
        app_state.cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        if app_state.gui:
            app_state.gui.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()