#!/usr/bin/env python3
"""
Quick start script for DeepTrix-Z.
Installs dependencies and runs a demo.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install required Python dependencies from requirements.txt."""
    print("📦 Installing required dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please try installing the dependencies manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)


def run_demo():
    """Run the main demo of the Tetris bot."""
    print("\n🚀 Running DeepTrix-Z demo...")
    try:
        # Check if main.py exists
        if not os.path.exists('main.py'):
            print("❌ 'main.py' not found. Make sure you are in the project root directory.")
            sys.exit(1)
        
        # Run the demo
        subprocess.check_call([sys.executable, "main.py", "demo"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed to run: {e}")
        print("Please check the console output for errors.")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Could not find 'python' or 'python3'. Make sure Python is in your PATH.")
        sys.exit(1)


def main():
    """Main function to run the quick start script."""
    print("=" * 60)
    print("🚀 Welcome to DeepTrix-Z Quick Start!")
    print("=" * 60)
    
    # Check for Python version
    if sys.version_info < (3, 9):
        print(f"❌ Your Python version is {sys.version_info.major}.{sys.version_info.minor}.")
        print("DeepTrix-Z requires Python 3.9 or higher.")
        sys.exit(1)
    
    # Check if we are in a virtual environment
    if "VIRTUAL_ENV" not in os.environ:
        print("⚠️  Warning: You are not in a Python virtual environment.")
        print("It is highly recommended to use a virtual environment to avoid package conflicts.")
        print("You can create one with: python3 -m venv venv && source venv/bin/activate")
        
        # Ask user if they want to continue
        choice = input("Do you want to continue without a virtual environment? (y/n): ")
        if choice.lower() != 'y':
            print("Aborting.")
            sys.exit(0)
    
    # Install dependencies
    install_dependencies()
    
    # Run demo
    run_demo()
    
    print("\n" + "=" * 60)
    print("🎉 Quick start complete! The bot is ready to go.")
    print("To train the model, run: python main.py train")
    print("To evaluate a model, run: python main.py evaluate --model <model_path>")
    print("=" * 60)


if __name__ == "__main__":
    main() 