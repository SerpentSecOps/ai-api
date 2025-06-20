#!/usr/bin/env python3
"""
Setup script for Local LLM Server
"""

import sys
import subprocess
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "llama-cpp-python>=0.2.0",
        "psutil>=5.9.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {req}: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "exports", "models"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Failed to create directory {directory}: {e}")

def main():
    """Main setup function"""
    print("Setting up Local LLM Server...")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if install_requirements():
        print("\nSetup completed successfully!")
        print("Run 'python main.py' to start the application.")
    else:
        print("\nSetup failed. Please install requirements manually.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())