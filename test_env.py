"""
AirCanvas AI - Environment Verification Script
Validates that OpenCV, MediaPipe, and NumPy are installed correctly in the virtual environment.
"""

import sys
import platform

def main():
    print("=" * 50)
    print("AirCanvas AI - Environment Verification")
    print("=" * 50)
    print(f"Python:       {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Executable:   {sys.executable}")
    print("-" * 50)

    all_passed = True

    # 1. Test OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV:    {cv2.__version__}")
    except Exception as e:
        print(f"[FAIL] OpenCV import failed: {e}")
        all_passed = False

    # 2. Test MediaPipe
    try:
        import mediapipe as mp
        print(f"[OK] MediaPipe: {mp.__version__}")
    except Exception as e:
        print(f"[FAIL] MediaPipe import failed: {e}")
        all_passed = False

    # 3. Test NumPy
    try:
        import numpy as np
        print(f"[OK] NumPy:     {np.__version__}")
    except Exception as e:
        print(f"[FAIL] NumPy import failed: {e}")
        all_passed = False

    print("=" * 50)
    if all_passed:
        print("STATUS: All required packages imported successfully!")
        sys.exit(0)
    else:
        print("STATUS: Verification failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
