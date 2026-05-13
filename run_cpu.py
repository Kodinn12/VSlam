#!/usr/bin/env python3
"""Run SLAM in CPU-only mode."""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run SLAM with CPU configuration."""
    print("=== Starting SLAM System (CPU Mode) ===")
    
    # Set config file for CPU mode
    config_file = 'config/cpu_config.yaml'
    
    if not os.path.exists(config_file):
        print(f"CPU config file not found: {config_file}")
        sys.exit(1)
    
    print(f"Using CPU config: {config_file}")
    
    # Override config in main module
    import src.main as main_module
    original_config = getattr(main_module, 'config_file', None)
    main_module.config_file = config_file
    
    # Import and run SLAM
    try:
        from src.main import main as slam_main
        slam_main()
    except KeyboardInterrupt:
        print("\nSLAM stopped by user")
    except Exception as e:
        print(f"SLAM error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
