"""Entry point for the SLAM system."""

# import yaml  # DISABLED - File saving removed per user request
import argparse
import cv2
import time
import sys
import numpy as np
import yaml
from src.slam_system import RobustStereoSLAM, TrackingState
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main(args_list=None):
    parser = argparse.ArgumentParser(description="OAK-D Stereo SLAM")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(args_list)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)

    slam = RobustStereoSLAM(config)
    print("\n SLAM running - press q to quit\n")

    try:
        while True:
            pose = slam.process_frame()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                np.save(f"slam_pose_{ts}.npy", pose)
                logger.info(f"Snapshot saved: slam_pose_{ts}.npy")
            elif key == ord('r'):
                logger.info("Manual relocalization triggered")
                slam.state = TrackingState.RELOCALIZING
                slam.relocalizer.scatter_hypotheses(slam.curr_pose, num_particles=256, mode="hybrid")
            elif key == ord('v'):
                config["show_voxels"] = not config.get("show_voxels", True)
                logger.info(f"Voxel vis: {'ON' if config['show_voxels'] else 'OFF'}")
            elif key == ord('b'):
                config["show_bubbles"] = not config.get("show_bubbles", True)
                logger.info(f"Bubble vis: {'ON' if config['show_bubbles'] else 'OFF'}")
            elif key == ord('f'):
                config["force_bubble_update"] = not config.get("force_bubble_update", False)
                logger.info(f"Force bubble update: {'ON' if config['force_bubble_update'] else 'OFF'}")
            elif key == ord('c'):
                if slam.visualizer:
                    slam.visualizer.trajectory_points = []
                    logger.info("Cleared 3D trajectory")
            if slam.visualizer and not slam.visualizer.is_active():
                logger.info("3D window closed, continuing without 3D view")
                slam.visualizer = None
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        slam.shutdown()
        logger.info("Application terminated")

if __name__ == "__main__":
    main()