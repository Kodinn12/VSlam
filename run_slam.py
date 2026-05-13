#!/usr/bin/env python
"""Top-level entry point for SLAM system (avoids relative import issues)."""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main
from main import main

if __name__ == '__main__':
    main()
