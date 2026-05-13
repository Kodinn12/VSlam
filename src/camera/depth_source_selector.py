"""Runtime depth source selector for OAK-D SLAM system."""

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DepthSourceSelector:
    """
    Runtime depth source selector for OAK-D SLAM system.
    
    Manages selection between OAK-D hardware depth and custom SGM
    depth sources based on configuration.
    """
    
    VALID_SOURCES = ['oakd_hardware', 'custom_sgm', 'ultimate_stereo']
    
    def __init__(self, config: dict):
        """
        Initialize depth source selector.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing depth_source parameter
        """
        self.config = config
        self.current_source = self._validate_and_get_source(config.get('depth_source', 'oakd_hardware'))
        logger.info(f"Depth source selector initialized with source: {self.current_source}")
    
    def _validate_and_get_source(self, source: str) -> str:
        """
        Validate and return depth source.
        
        Parameters
        ----------
        source : str
            Requested depth source
        
        Returns
        -------
        str
            Validated depth source
        """
        if source not in self.VALID_SOURCES:
            logger.warning(f"Invalid depth source '{source}', defaulting to 'oakd_hardware'")
            return 'oakd_hardware'
        return source
    
    def get_source(self) -> str:
        """
        Get current depth source.
        
        Returns
        -------
        str
            Current depth source ('oakd_hardware' or 'custom_sgm')
        """
        return self.current_source
    
    def is_hardware_depth(self) -> bool:
        """
        Check if current source is OAK-D hardware depth.
        
        Returns
        -------
        bool
            True if using OAK-D hardware depth
        """
        return self.current_source == 'oakd_hardware'
    
    def is_custom_sgm(self) -> bool:
        """
        Check if current source is custom SGM.
        
        Returns
        -------
        bool
            True if using custom SGM
        """
        return self.current_source == 'custom_sgm'
    
    def switch_source(self, new_source: str) -> bool:
        """
        Switch to a different depth source (requires pipeline rebuild).
        
        Parameters
        ----------
        new_source : str
            New depth source to switch to
        
        Returns
        -------
        bool
            True if switch was successful
        """
        if new_source not in self.VALID_SOURCES:
            logger.error(f"Invalid depth source '{new_source}'")
            return False
        
        if new_source == self.current_source:
            logger.info(f"Already using depth source '{new_source}'")
            return True
        
        logger.info(f"Switching depth source from '{self.current_source}' to '{new_source}'")
        logger.warning("Note: Switching depth source requires pipeline rebuild")
        self.current_source = new_source
        return True
