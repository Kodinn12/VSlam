"""Lazy mirrors system for efficient GPU memory management."""

import numpy as np
import threading
from collections import OrderedDict
from ..utils.logger import get_logger

logger = get_logger(__name__)

# CUDA availability check
try:
    import cupy as cp
    USE_CUPY = True
    logger.info("CuPy available for lazy mirrors")
except ImportError:
    cp = None
    USE_CUPY = False
    logger.info("CuPy not available, lazy mirrors disabled")

class LazyMirror:
    """Lazy mirror for CPU arrays with on-demand GPU synchronization."""
    
    def __init__(self, cpu_array, manager, name=None):
        self.cpu_array = cpu_array
        self.manager = manager
        self.name = name
        self.gpu_array = None
        self.gpu_version = 0
        self.cpu_version = 1
        self.dirty_cpu = True
        self.dirty_gpu = False
        self.lock = threading.RLock()
        self.access_count = 0
        self.last_access_time = 0
        self.size = cpu_array.nbytes if hasattr(cpu_array, 'nbytes') else 0
        
    def get_gpu_array(self, force_sync=False):
        """Get GPU array, synchronizing from CPU if needed."""
        if not USE_CUPY:
            return self.cpu_array
        
        with self.lock:
            self.access_count += 1
            self.last_access_time = self.manager.get_current_time()
            
            if self.gpu_array is None or force_sync or self.dirty_cpu:
                # Transfer to GPU
                self.gpu_array = cp.asarray(self.cpu_array)
                self.gpu_version = self.cpu_version
                self.dirty_cpu = False
                self.dirty_gpu = False
                
                # Update manager statistics
                self.manager.record_gpu_transfer(self.size)
            
            return self.gpu_array
    
    def get_cpu_array(self, force_sync=False):
        """Get CPU array, synchronizing from GPU if needed."""
        with self.lock:
            self.access_count += 1
            self.last_access_time = self.manager.get_current_time()
            
            if self.dirty_gpu or force_sync:
                # Transfer from GPU
                if self.gpu_array is not None:
                    self.cpu_array[:] = cp.asnumpy(self.gpu_array)
                    self.cpu_version = self.gpu_version
                    self.dirty_cpu = False
                    self.dirty_gpu = False
                    
                    # Update manager statistics
                    self.manager.record_cpu_transfer(self.size)
            
            return self.cpu_array
    
    def mark_cpu_dirty(self):
        """Mark CPU array as modified."""
        with self.lock:
            self.cpu_version += 1
            self.dirty_cpu = True
            self.dirty_gpu = False
    
    def mark_gpu_dirty(self):
        """Mark GPU array as modified."""
        with self.lock:
            self.gpu_version += 1
            self.dirty_gpu = True
            self.dirty_cpu = False
    
    def synchronize_to_gpu(self):
        """Force synchronization to GPU."""
        self.get_gpu_array(force_sync=True)
    
    def synchronize_to_cpu(self):
        """Force synchronization to CPU."""
        self.get_cpu_array(force_sync=True)
    
    def invalidate_gpu(self):
        """Invalidate GPU array to save memory."""
        with self.lock:
            if self.gpu_array is not None:
                del self.gpu_array
                self.gpu_array = None
                self.dirty_cpu = True
    
    def invalidate_cpu(self):
        """Invalidate CPU array to save memory."""
        with self.lock:
            if self.dirty_gpu:
                # Save GPU data before invalidating CPU
                self.cpu_array = cp.asnumpy(self.gpu_array)
                self.cpu_version = self.gpu_version
                self.dirty_cpu = False
                self.dirty_gpu = False
    
    def get_memory_usage(self):
        """Get memory usage statistics."""
        with self.lock:
            return {
                'cpu_size': self.size,
                'gpu_size': self.gpu_array.nbytes if self.gpu_array is not None else 0,
                'access_count': self.access_count,
                'last_access': self.last_access_time,
                'dirty_cpu': self.dirty_cpu,
                'dirty_gpu': self.dirty_gpu,
                'has_gpu': self.gpu_array is not None
            }

class LazyMirrorManager:
    """Manages lazy mirrors with LRU eviction and memory optimization."""
    
    def __init__(self, max_gpu_memory=1024*1024*1024, max_mirrors=1000):
        self.max_gpu_memory = max_gpu_memory  # 1GB default
        self.max_mirrors = max_mirrors
        self.mirrors = OrderedDict()  # LRU cache
        self.mirror_stats = {}
        self.current_gpu_memory = 0
        self.lock = threading.RLock()
        self.access_counter = 0
        self.start_time = self.get_current_time()
        
    def get_current_time(self):
        """Get current time in seconds."""
        import time
        return time.time()
    
    def create_mirror(self, cpu_array, name=None):
        """Create lazy mirror for CPU array."""
        if not USE_CUPY:
            return cpu_array
        
        with self.lock:
            # Check if mirror already exists
            array_id = id(cpu_array)
            if array_id in self.mirrors:
                return self.mirrors[array_id]
            
            # Create new mirror
            mirror = LazyMirror(cpu_array, self, name)
            
            # Add to LRU cache
            self.mirrors[array_id] = mirror
            
            # Update statistics
            self.mirror_stats[name or f"mirror_{array_id}"] = {
                'created_at': self.get_current_time(),
                'size': mirror.size
            }
            
            # Evict if necessary
            self._evict_if_needed()
            
            return mirror
    
    def get_mirror(self, cpu_array):
        """Get existing mirror for CPU array."""
        with self.lock:
            array_id = id(cpu_array)
            return self.mirrors.get(array_id)
    
    def remove_mirror(self, cpu_array):
        """Remove mirror for CPU array."""
        with self.lock:
            array_id = id(cpu_array)
            if array_id in self.mirrors:
                mirror = self.mirrors[array_id]
                self.current_gpu_memory -= mirror.get_memory_usage()['gpu_size']
                del self.mirrors[array_id]
                mirror.invalidate_gpu()
    
    def _evict_if_needed(self):
        """Evict mirrors if memory or count limits exceeded."""
        # Evict by memory usage
        while (self.current_gpu_memory > self.max_gpu_memory and 
               len(self.mirrors) > 0):
            self._evict_lru()
        
        # Evict by count
        while len(self.mirrors) > self.max_mirrors:
            self._evict_lru()
    
    def _evict_lru(self):
        """Evict least recently used mirror."""
        if not self.mirrors:
            return
        
        # Get LRU mirror
        lru_id, lru_mirror = next(iter(self.mirrors.items()))
        
        # Invalidate GPU array
        gpu_size = lru_mirror.get_memory_usage()['gpu_size']
        lru_mirror.invalidate_gpu()
        
        # Update memory usage
        self.current_gpu_memory -= gpu_size
        
        # Move to end of OrderedDict (LRU)
        self.mirrors.move_to_end(lru_id)
    
    def record_gpu_transfer(self, size):
        """Record GPU memory transfer."""
        with self.lock:
            self.current_gpu_memory += size
            self.access_counter += 1
            self._evict_if_needed()
    
    def record_cpu_transfer(self, size):
        """Record CPU memory transfer."""
        with self.lock:
            self.access_counter += 1
    
    def prefetch_mirrors(self, mirror_names=None, max_memory=None):
        """Prefetch mirrors to GPU."""
        if not USE_CUPY:
            return
        
        with self.lock:
            if mirror_names is None:
                # Prefetch most recently used mirrors
                mirrors_to_prefetch = list(self.mirrors.values())[-10:]
            else:
                mirrors_to_prefetch = [m for m in self.mirrors.values() 
                                     if m.name in mirror_names]
            
            memory_budget = max_memory or (self.max_gpu_memory - self.current_gpu_memory)
            
            for mirror in mirrors_to_prefetch:
                if memory_budget <= 0:
                    break
                
                memory_usage = mirror.get_memory_usage()
                if memory_usage['gpu_size'] == 0:  # Not on GPU yet
                    mirror.get_gpu_array()
                    memory_budget -= memory_usage['gpu_size']
    
    def cleanup_unused_mirrors(self, max_idle_time=300):
        """Clean up mirrors that haven't been accessed recently."""
        current_time = self.get_current_time()
        
        with self.lock:
            to_remove = []
            for mirror_id, mirror in self.mirrors.items():
                memory_usage = mirror.get_memory_usage()
                if (current_time - memory_usage['last_access'] > max_idle_time and
                    memory_usage['access_count'] > 0):
                    to_remove.append(mirror_id)
            
            for mirror_id in to_remove:
                self.remove_mirror(self.mirrors[mirror_id].cpu_array)
            
            logger.info(f"Cleaned up {len(to_remove)} unused mirrors")
    
    def get_statistics(self):
        """Get mirror manager statistics."""
        with self.lock:
            stats = {
                'total_mirrors': len(self.mirrors),
                'current_gpu_memory': self.current_gpu_memory,
                'max_gpu_memory': self.max_gpu_memory,
                'access_counter': self.access_counter,
                'uptime': self.get_current_time() - self.start_time,
                'memory_efficiency': self.current_gpu_memory / self.max_gpu_memory
            }
            
            # Per-mirror statistics
            mirror_stats = {}
            for mirror_id, mirror in self.mirrors.items():
                mirror_stats[mirror.name or f"mirror_{mirror_id}"] = mirror.get_memory_usage()
            
            stats['mirrors'] = mirror_stats
            return stats
    
    def optimize_memory(self):
        """Optimize memory usage by cleaning up unused mirrors."""
        self.cleanup_unused_mirrors()
        
        # Force garbage collection
        if USE_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
        
        logger.info("Memory optimization complete")
    
    def shutdown(self):
        """Shutdown mirror manager."""
        with self.lock:
            # Invalidate all GPU arrays
            for mirror in self.mirrors.values():
                mirror.invalidate_gpu()
            
            # Clear mirrors
            self.mirrors.clear()
            self.current_gpu_memory = 0
            
            logger.info("Lazy mirror manager shutdown complete")

# Global lazy mirror manager instance
lazy_mirror_manager = LazyMirrorManager()
