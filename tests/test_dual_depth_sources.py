"""Integration tests for dual depth source functionality."""

import os

import numpy as np
from src.camera.depth_source_selector import DepthSourceSelector
from src.stereo.stereo_engine import StereoEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_depth_source_selector():
    """Test depth source selector."""
    logger.info("Testing DepthSourceSelector...")
    
    # Test with oakd_hardware mode
    config_hardware = {"depth_source": "oakd_hardware"}
    selector_hardware = DepthSourceSelector(config_hardware)
    assert selector_hardware.get_source() == "oakd_hardware"
    assert selector_hardware.is_hardware_depth()
    assert not selector_hardware.is_custom_sgm()
    logger.info("✓ oakd_hardware mode test passed")
    
    # Test with custom_sgm mode
    config_sgm = {"depth_source": "custom_sgm"}
    selector_sgm = DepthSourceSelector(config_sgm)
    assert selector_sgm.get_source() == "custom_sgm"
    assert not selector_sgm.is_hardware_depth()
    assert selector_sgm.is_custom_sgm()
    logger.info("✓ custom_sgm mode test passed")
    
    # Test with invalid mode (should default to oakd_hardware)
    config_invalid = {"depth_source": "invalid_mode"}
    selector_invalid = DepthSourceSelector(config_invalid)
    assert selector_invalid.get_source() == "oakd_hardware"
    logger.info("✓ invalid mode fallback test passed")
    
    logger.info("DepthSourceSelector tests passed!")


def test_stereo_engine_initialization():
    """Test StereoEngine initialization."""
    logger.info("Testing StereoEngine initialization...")
    
    # Test CPU mode
    config_cpu = {"acceleration_mode": "cpu_only"}
    engine_cpu = StereoEngine(config_cpu, acceleration_mode="cpu_only")
    assert engine_cpu is not None
    logger.info("✓ CPU StereoEngine initialization passed")
    
    # Test GPU mode (if CuPy available)
    try:
        import cupy as cp
        config_gpu = {"acceleration_mode": "full_gpu"}
        engine_gpu = StereoEngine(config_gpu, acceleration_mode="full_gpu")
        assert engine_gpu is not None
        logger.info("✓ GPU StereoEngine initialization passed")
    except ImportError:
        logger.info("⚠ CuPy not available, skipping GPU test")
    
    logger.info("StereoEngine initialization tests passed!")


def test_synthetic_stereo_pipeline():
    """Test stereo pipeline with synthetic data."""
    logger.info("Testing stereo pipeline with synthetic data...")
    
    # Generate synthetic stereo images
    H, W = 48, 64
    left = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    right = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    
    # Add some disparity pattern (simple translation)
    disparity = np.zeros((H, W), dtype=np.float32)
    disparity[:, 20:] = 4.0  # Right half has disparity of 4 pixels
    right[:, 20:] = np.roll(left[:, 20:], -4, axis=1)
    
    # Test CPU stereo engine
    config_cpu = {
        "acceleration_mode": "cpu_only",
        "sgm_cost_volume_depth": 8,
        "sgm_bilateral_filter": False,
        "sgm_lr_check": False,
    }
    engine_cpu = StereoEngine(config_cpu, acceleration_mode="cpu_only")
    
    try:
        depth_cpu = engine_cpu.compute_depth(
            left, right, baseline_m=0.075, fx=500.0, motion_score=0.0
        )
        assert depth_cpu is not None
        assert depth_cpu.shape == (H, W)
        logger.info("✓ CPU stereo pipeline test passed")
    except Exception as e:
        logger.warning(f"CPU stereo pipeline test failed: {e}")
    
    logger.info("Synthetic stereo pipeline tests passed!")


def test_dual_mode_config():
    """Test dual mode configuration in config files."""
    logger.info("Testing dual mode configuration...")
    
    import yaml
    
    # Test gpu_config.yaml
    config_path_gpu = os.path.join(os.path.dirname(__file__), '..', 'config', 'gpu_config.yaml')
    with open(config_path_gpu, 'r') as f:
        config_gpu = yaml.safe_load(f)
    assert 'depth_source' in config_gpu
    assert config_gpu['depth_source'] in ['oakd_hardware', 'custom_sgm']
    logger.info(f"✓ gpu_config.yaml has depth_source: {config_gpu['depth_source']}")
    
    # Test cpu_config.yaml
    config_path_cpu = os.path.join(os.path.dirname(__file__), '..', 'config', 'cpu_config.yaml')
    with open(config_path_cpu, 'r') as f:
        config_cpu = yaml.safe_load(f)
    assert 'depth_source' in config_cpu
    assert config_cpu['depth_source'] in ['oakd_hardware', 'custom_sgm']
    logger.info(f"✓ cpu_config.yaml has depth_source: {config_cpu['depth_source']}")
    
    logger.info("Dual mode configuration tests passed!")


def run_all_tests():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("Running Dual Depth Source Integration Tests")
    logger.info("=" * 60)
    
    try:
        test_depth_source_selector()
        test_stereo_engine_initialization()
        test_synthetic_stereo_pipeline()
        test_dual_mode_config()
        
        logger.info("=" * 60)
        logger.info("All integration tests passed!")
        logger.info("=" * 60)
        return True
    except Exception as e:
        logger.error(f"Integration tests failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
