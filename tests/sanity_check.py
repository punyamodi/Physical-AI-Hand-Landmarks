import sys
import os

def test_imports():
    try:
        import cv2
        import mediapipe
        import pandas
        import numpy
        import customtkinter
        import PIL
        print("✅ All imports successful.")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    return True

def test_collector():
    try:
        from src.collector import HandDataCollector
        import os
        model_path = os.path.join("src", "hand_landmarker.task")
        collector = HandDataCollector(model_path=model_path)
        print("✅ Colletor initialized with Tasks API.")
    except Exception as e:
        print(f"❌ Collector initialization failed: {e}")
        return False
    return True

if __name__ == "__main__":
    s1 = test_imports()
    s2 = test_collector()
    if s1 and s2:
        print("🚀 Sanity check passed!")
        sys.exit(0)
    else:
        print("❌ Sanity check failed!")
        sys.exit(1)
