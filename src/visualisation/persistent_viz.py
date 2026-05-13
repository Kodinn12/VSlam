"""Persistent PyVista visualization for stable global map rendering."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pyvista as pv
    USE_PYVISTA = True
except ImportError:
    USE_PYVISTA = False
    logger.warning("PyVista not available, persistent visualization disabled")

class PersistentVisualizer:
    """
    Persistent visualizer that accumulates points and provides stable global map rendering.
    
    Features:
    - Point accumulation across frames
    - Performance-optimized downsampling
    - Consistent full-scene visualization
    - No view-dependent filtering
    """
    
    def __init__(self, max_display_points=300000):
        """
        Initialize persistent visualizer.
        
        Args:
            max_display_points: Maximum points to display for performance
        """
        if not USE_PYVISTA:
            logger.error("PyVista not available, persistent visualizer disabled")
            return
        
        self.pv = pv
        self.max_display_points = max_display_points
        
        # Point storage
        self.global_points = None
        self.global_colors = None
        
        # PyVista components
        self.plotter = None
        self.points_actor = None
        
        # Initialize plotter
        self._setup_plotter()
        
        logger.info(f"Persistent Visualizer initialized (max_display={max_display_points})")
    
    def _setup_plotter(self):
        """Setup PyVista plotter with unified coordinate system."""
        try:
            self.plotter = self.pv.Plotter()
            self.plotter.set_background('#0a0a0a') # Deep dark
            self.plotter.add_axes()
            self.plotter.add_camera_orientation_widget()
            
            # Use same initial camera as other visualizers: isometric behind/above
            self.plotter.camera_position = [(0, -5, 3), (0, 0, 0), (0, 0, 1)]
            self.plotter.show(interactive_update=True, auto_close=False)
        except Exception as e:
            logger.error(f"Failed to setup PyVista plotter: {e}")
    
    def _to_pyvista(self, points):
        """Standard project transform: OAK-D (X,Y,Z) -> PyVista (X,Z,-Y)."""
        if points is None: return None
        out = points.copy()
        out[:, 1] = points[:, 2]  # Y' = Z (forward)
        out[:, 2] = -points[:, 1] # Z' = -Y (up)
        return out

    def update(self, points, colors=None):
        """
        Update visualization with new points.
        """
        if not USE_PYVISTA or self.plotter is None:
            return
        
        if len(points) == 0:
            return
        
        try:
            # Apply standard coordinate transformation
            points_pv = self._to_pyvista(points)

            # Initialize storage if needed
            if self.global_points is None:
                self.global_points = points_pv
                self.global_colors = colors.copy() if colors is not None else np.ones_like(points)
            else:
                # Accumulate points
                self.global_points = np.vstack([self.global_points, points_pv])
                if colors is not None:
                    self.global_colors = np.vstack([self.global_colors, colors])
                else:
                    new_colors = np.ones((len(points), 3))
                    self.global_colors = np.vstack([self.global_colors, new_colors])
            
            # Downsample for performance if needed
            if self.global_points.shape[0] > self.max_display_points:
                display_points, display_colors = self._downsample_points(
                    self.global_points, self.global_colors, self.max_display_points
                )
            else:
                display_points = self.global_points
                display_colors = self.global_colors
            
            # Create point cloud
            cloud = self.pv.PolyData(display_points)
            if display_colors is not None:
                if display_colors.max() > 1.01:
                    display_colors = display_colors / 255.0
                cloud['color'] = display_colors
            
            # Update or create actor
            if self.points_actor is None:
                self.points_actor = self.plotter.add_points(
                    cloud, 
                    scalars='color',
                    rgb=True,
                    render_points_as_spheres=False, # Sharper flat discs
                    point_size=2.0
                )
            else:
                self.points_actor.mapper.SetInputData(cloud)
            
            self.plotter.render()
            
        except Exception as e:
            logger.error(f"Failed to update persistent visualization: {e}")
    
    def _downsample_points(self, points, colors, target_count):
        """
        Downsample points while maintaining spatial distribution.
        
        Args:
            points: (N, 3) input points
            colors: (N, 3) input colors  
            target_count: Target number of points
            
        Returns:
            (points_downsampled, colors_downsampled)
        """
        if len(points) <= target_count:
            return points, colors
        
        # Simple random downsampling for now
        # Could implement more sophisticated spatial downsampling later
        indices = np.random.choice(len(points), target_count, replace=False)
        return points[indices], colors[indices] if colors is not None else None
    
    def clear(self):
        """Clear all accumulated points."""
        self.global_points = None
        self.global_colors = None
        if self.points_actor is not None and self.plotter is not None:
            self.plotter.remove_actor(self.points_actor)
            self.points_actor = None
        logger.info("Persistent visualization cleared")
    
    def get_statistics(self):
        """Get visualization statistics."""
        return {
            'total_points': len(self.global_points) if self.global_points is not None else 0,
            'display_points': min(len(self.global_points) if self.global_points is not None else 0, 
                                 self.max_display_points),
            'max_display_points': self.max_display_points
        }
    
    def close(self):
        """Close the visualizer."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None
        logger.info("Persistent visualizer closed")

def create_persistent_visualizer(max_display_points=300000):
    """
    Factory function to create persistent visualizer.
    
    Args:
        max_display_points: Maximum points to display
        
    Returns:
        PersistentVisualizer instance or None if unavailable
    """
    if USE_PYVISTA:
        return PersistentVisualizer(max_display_points)
    else:
        logger.warning("PyVista not available, cannot create persistent visualizer")
        return None
