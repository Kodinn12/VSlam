"""Profiling utilities for per-module timing and bottleneck detection."""

import time
from functools import wraps
from collections import defaultdict
from typing import Dict, List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Profiler:
    """Profiler for per-function timing and bottleneck detection."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize profiler.
        
        Parameters
        ----------
        enabled : bool
            Whether profiling is enabled
        """
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self._current_starts: Dict[str, float] = {}
    
    def start(self, name: str):
        """
        Start timing a section.
        
        Parameters
        ----------
        name : str
            Section name
        """
        if not self.enabled:
            return
        self._current_starts[name] = time.perf_counter()
    
    def end(self, name: str) -> float:
        """
        End timing a section and record the duration.
        
        Parameters
        ----------
        name : str
            Section name
        
        Returns
        -------
        float
            Duration in seconds
        """
        if not self.enabled or name not in self._current_starts:
            return 0.0
        
        duration = time.perf_counter() - self._current_starts[name]
        del self._current_starts[name]
        
        self.timings[name].append(duration)
        self.call_counts[name] += 1
        
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a section.
        
        Parameters
        ----------
        name : str
            Section name
        
        Returns
        -------
        dict
            Statistics (mean, min, max, total, count)
        """
        if name not in self.timings or len(self.timings[name]) == 0:
            return {}
        
        timings = self.timings[name]
        return {
            'mean': sum(timings) / len(timings),
            'min': min(timings),
            'max': max(timings),
            'total': sum(timings),
            'count': len(timings)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all sections.
        
        Returns
        -------
        dict
            Dictionary of section statistics
        """
        return {name: self.get_stats(name) for name in self.timings.keys()}
    
    def print_summary(self, top_n: int = 10):
        """
        Print profiling summary.
        
        Parameters
        ----------
        top_n : int
            Number of top sections to show
        """
        if not self.enabled:
            logger.info("Profiling is disabled")
            return
        
        stats = self.get_all_stats()
        
        # Sort by total time
        sorted_sections = sorted(stats.items(), key=lambda x: x[1].get('total', 0), reverse=True)
        
        logger.info("=" * 60)
        logger.info("Profiling Summary")
        logger.info("=" * 60)
        
        for i, (name, stat) in enumerate(sorted_sections[:top_n]):
            logger.info(f"{i+1:2d}. {name:30s} "
                       f"total={stat['total']*1000:7.2f}ms "
                       f"mean={stat['mean']*1000:7.2f}ms "
                       f"count={stat['count']:5d}")
        
        logger.info("=" * 60)
    
    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()
        self.call_counts.clear()
        self._current_starts.clear()


# Global profiler instance
_global_profiler = Profiler(enabled=False)


def profile(name: str = None):
    """
    Decorator for profiling function execution time.
    
    Parameters
    ----------
    name : str
        Section name (defaults to function name)
    
    Returns
    -------
    function
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            section_name = name or func.__name__
            _global_profiler.start(section_name)
            result = func(*args, **kwargs)
            _global_profiler.end(section_name)
            return result
        return wrapper
    return decorator


def enable_profiling():
    """Enable global profiling."""
    _global_profiler.enabled = True
    logger.info("Profiling enabled")


def disable_profiling():
    """Disable global profiling."""
    _global_profiler.enabled = False
    logger.info("Profiling disabled")


def get_profiler() -> Profiler:
    """
    Get global profiler instance.
    
    Returns
    -------
    Profiler
        Global profiler
    """
    return _global_profiler


def print_profiling_summary(top_n: int = 10):
    """
    Print global profiling summary.
    
    Parameters
    ----------
    top_n : int
        Number of top sections to show
    """
    _global_profiler.print_summary(top_n)


def reset_profiling():
    """Reset global profiling data."""
    _global_profiler.reset()
