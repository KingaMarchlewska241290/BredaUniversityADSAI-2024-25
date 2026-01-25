"""
Simple file check - verifies all files are present
No imports needed
"""

from pathlib import Path

print("="*60)
print("ASL Recognition - File Check")
print("="*60)

current_dir = Path.cwd()
print(f"\nCurrent directory: {current_dir}")

# Files that should exist:
files_to_check = {
    'Main Scripts': [
        'main.py',
        'asl_recognizer.py', 
        'test_system.py',
        'requirements.txt'
    ],
    'Utils': [
        'utils/__init__.py',
        'utils/hand_detector.py',
        'utils/dynamic_classifier.py'
    ],
    'Models': [
        'models/final_model_1.h5',
        'models/class_indices_1.json'
    ],
    'Documentation': [
        'README_QUICK_SETUP.md'
    ]
}

all_present = True
for category, files in files_to_check.items():
    print(f"\n{category}:")
    for file in files:
        file_path = current_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"{file} ({size:,} bytes)")
        else:
            print(f"{file} - MISSING")
            all_present = False

print("\n" + "="*60)
if all_present:
    print("All files present!")
    print("\nNext steps:")
    print("1. Run: python check_environment.py")
    print("2. Then: python test_system.py")
else:
    print("Some files are missing")
    print("Make sure you extracted the complete asl_recognition folder")
print("="*60)
