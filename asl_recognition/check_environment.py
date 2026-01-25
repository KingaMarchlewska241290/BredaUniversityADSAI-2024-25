"""
Diagnostic script to check if your environment is set up correctly
Run this if you have import errors
"""

import sys
import os
from pathlib import Path

print("="*60)
print("ASL Recognition - Environment Diagnostic")
print("="*60)

# Checking Python version:
print(f"\n1. Python Version: {sys.version}")
python_version = sys.version_info
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("Python 3.8+ recommended")
else:
    print("Python version OK")

# Checking current directory:
current_dir = Path.cwd()
print(f"\n2. Current Directory: {current_dir}")

# Checking if in the right directory:
expected_files = ['main.py', 'asl_recognizer.py', 'test_system.py', 'requirements.txt']
missing_files = [f for f in expected_files if not (current_dir / f).exists()]

if missing_files:
    print(f"Missing files: {missing_files}")
    print("Make sure you're in the 'asl_recognition' directory")
else:
    print("All main files present")

# Checking directories:
print(f"\n3. Directory Structure:")
utils_dir = current_dir / 'utils'
models_dir = current_dir / 'models'

if utils_dir.exists():
    utils_files = list(utils_dir.glob('*.py'))
    print(f"utils/ directory exists ({len(utils_files)} Python files)")
    for f in utils_files:
        print(f"      - {f.name}")
else:
    print("utils/ directory NOT FOUND")

if models_dir.exists():
    model_files = list(models_dir.glob('*'))
    print(f"models/ directory exists ({len(model_files)} files)")
    for f in model_files:
        print(f"- {f.name}")
else:
    print("models/ directory NOT FOUND")

# Checking required packages:
print(f"\n4. Checking Required Packages:")
required_packages = {
    'tensorflow': 'Model inference',
    'cv2': 'opencv-python (Camera and image processing)',
    'mediapipe': 'Hand detection',
    'numpy': 'Array operations'
}

all_installed = True
for package_name, description in required_packages.items():
    try:
        if package_name == 'cv2':
            import cv2
            print(f"opencv-python: {cv2.__version__}")
        elif package_name == 'tensorflow':
            import tensorflow as tf
            print(f"tensorflow: {tf.__version__}")
        elif package_name == 'mediapipe':
            import mediapipe as mp
            print(f"mediapipe: {mp.__version__}")
        elif package_name == 'numpy':
            import numpy as np
            print(f"numpy: {np.__version__}")
    except ImportError:
        print(f"{description} ({package_name}) - NOT INSTALLED")
        all_installed = False

# Checking if imports work:
print(f"\n5. Testing Module Imports:")
try:
    # Adding current dir to path:
    sys.path.insert(0, str(current_dir))
    
    from utils.hand_detector import HandDetector
    print("utils.hand_detector")
    
    from utils.dynamic_classifier import DynamicLetterClassifier
    print("utils.dynamic_classifier")
    
    from asl_recognizer import ASLRecognizer
    print("asl_recognizer")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you're in the asl_recognition directory")
    print("2. Check that utils/__init__.py exists")
    print("3. Try: cd asl_recognition && python check_environment.py")

# Summary:
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if not missing_files and all_installed:
    print("Environment looks good! You can run:")
    print("python test_system.py")
    print("python main.py")
else:
    print("Issues found. Please fix the following:")
    if missing_files:
        print(f"- Missing files: {missing_files}")
    if not all_installed:
        print("- Install missing packages: pip install -r requirements.txt")
    if not utils_dir.exists():
        print("- utils/ directory is missing")
    if not models_dir.exists():
        print("- models/ directory is missing")

print("\n" + "="*60)
