import sys
import os

# Add current directory to path so 'src' is seen as a package
sys.path.insert(0, os.getcwd())

try:
    from src.camera.ultimate_stereo import UltimateStereoProcessor
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
