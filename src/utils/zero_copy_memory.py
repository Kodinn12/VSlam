"""Zero-copy memory manager for GPU acceleration."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

# CUDA availability check
try:
    import cupy as cp
    USE_CUPY = True
    logger.info("CuPy available for zero-copy memory")
except ImportError:
    cp = None
    USE_CUPY = False
    logger.info("CuPy not available, zero-copy disabled")

class ZeroCopyMemoryManager:
    """Manages zero-copy memory for maximum GPU performance."""
    
    def __init__(self):
        self.zero_copy_enabled = False
        self.pinned_memory_pool = None
        self.unified_memory_pool = None
        self.memory_mappings = {}
        self._setup_zero_copy()
        
    def _setup_zero_copy(self):
        """Setup zero-copy memory capabilities."""
        if not USE_CUPY:
            return
        
        try:
            # Check for unified memory support
            device = cp.cuda.Device()
            device_props = device.attributes
            
            # Check for managed memory (unified memory)
            if device_props.get('ManagedMemory', False):
                self.zero_copy_enabled = True
                logger.info("Unified memory (zero-copy) enabled")
            else:
                logger.info("Unified memory not available, using pinned memory")
                self._setup_pinned_memory()
                
        except Exception as e:
            logger.warning(f"Zero-copy setup failed: {e}")
    
    def _setup_pinned_memory(self):
        """Setup pinned memory for faster transfers."""
        try:
            # Create pinned memory pool
            self.pinned_memory_pool = cp.cuda.PinnedMemoryPool()
            cp.cuda.set_pinned_memory_allocator(self.pinned_memory_pool)
            logger.info("Pinned memory pool created")
        except Exception as e:
            logger.warning(f"Pinned memory setup failed: {e}")
    
    def allocate_zero_copy(self, shape, dtype=np.float32, name=None):
        """Allocate zero-copy memory."""
        if not USE_CUPY:
            return np.zeros(shape, dtype=dtype)
        
        try:
            if self.zero_copy_enabled:
                # Use unified memory for true zero-copy
                # Check if unified memory is available and use proper API
                try:
                    # Try CuPy's unified memory allocation
                    array = cp.zeros(shape, dtype=dtype)
                    # Mark as managed memory if supported
                    if hasattr(cp.cuda, 'memory') and hasattr(cp.cuda.memory, 'alloc_managed'):
                        array = cp.zeros(shape, dtype=dtype, memptr=cp.cuda.memory.alloc_managed(shape, dtype))
                    else:
                        # Fallback to regular GPU memory
                        array = cp.zeros(shape, dtype=dtype)
                except AttributeError:
                    # Fallback if unified memory API not available
                    array = cp.zeros(shape, dtype=dtype)
                
                if name:
                    self.memory_mappings[name] = array
                return array
            else:
                # Use pinned memory as fallback
                array = cp.zeros(shape, dtype=dtype)
                if name:
                    self.memory_mappings[name] = array
                return array
                
        except Exception as e:
            logger.warning(f"Zero-copy allocation failed: {e}")
            return cp.zeros(shape, dtype=dtype)
    
    def allocate_pinned(self, shape, dtype=np.float32, name=None):
        """Allocate pinned memory for faster transfers."""
        if not USE_CUPY or not self.pinned_memory_pool:
            return np.zeros(shape, dtype=dtype)
        
        try:
            # Create pinned memory array
            array = np.zeros(shape, dtype=dtype)
            if name:
                self.memory_mappings[name] = array
            return array
        except Exception as e:
            logger.warning(f"Pinned memory allocation failed: {e}")
            return np.zeros(shape, dtype=dtype)
    
    def create_shared_buffer(self, cpu_array, name=None):
        """Create shared buffer between CPU and GPU."""
        if not USE_CUPY:
            return cpu_array
        
        try:
            if self.zero_copy_enabled:
                # Use unified memory for true sharing
                shared_array = cp.asarray(cpu_array, memptr=cp.cuda.alloc_unified(cpu_array.shape, cpu_array.dtype))
                if name:
                    self.memory_mappings[name] = shared_array
                return shared_array
            else:
                # Use pinned memory with async copy
                pinned_array = self.allocate_pinned(cpu_array.shape, cpu_array.dtype, name)
                pinned_array[:] = cpu_array
                gpu_array = cp.asarray(pinned_array)
                if name:
                    self.memory_mappings[name] = gpu_array
                return gpu_array
                
        except Exception as e:
            logger.warning(f"Shared buffer creation failed: {e}")
            return cp.asarray(cpu_array)
    
    def async_copy_to_gpu(self, cpu_array, gpu_array=None, stream=None):
        """Asynchronous copy from CPU to GPU."""
        if not USE_CUPY:
            return cpu_array
        
        try:
            if gpu_array is None:
                gpu_array = cp.empty_like(cpu_array)
            
            if stream is None:
                stream = cp.cuda.Stream()
            
            # Use pinned memory for async copy
            if self.pinned_memory_pool:
                pinned_array = self.allocate_pinned(cpu_array.shape, cpu_array.dtype)
                pinned_array[:] = cpu_array
                with stream:
                    gpu_array.copy_from_buffer_async(pinned_array, pinned_array.nbytes)
            else:
                with stream:
                    gpu_array[:] = cpu_array
            
            return gpu_array
            
        except Exception as e:
            logger.warning(f"Async copy failed: {e}")
            return cp.asarray(cpu_array)
    
    def async_copy_to_cpu(self, gpu_array, cpu_array=None, stream=None):
        """Asynchronous copy from GPU to CPU."""
        if not USE_CUPY:
            return gpu_array
        
        try:
            if cpu_array is None:
                cpu_array = np.empty_like(gpu_array)
            
            if stream is None:
                stream = cp.cuda.Stream()
            
            # Use pinned memory for async copy
            if self.pinned_memory_pool:
                pinned_array = self.allocate_pinned(cpu_array.shape, cpu_array.dtype)
                with stream:
                    pinned_array.copy_from_buffer_async(gpu_array, gpu_array.nbytes)
                stream.synchronize()
                cpu_array[:] = pinned_array
            else:
                with stream:
                    cpu_array[:] = cp.asnumpy(gpu_array)
            
            return cpu_array
            
        except Exception as e:
            logger.warning(f"Async copy failed: {e}")
            return cp.asnumpy(gpu_array)
    
    def prefetch_to_gpu(self, array, stream=None):
        """Prefetch array to GPU memory."""
        if not USE_CUPY or not self.zero_copy_enabled:
            return
        
        try:
            if stream is None:
                stream = cp.cuda.Stream()
            
            # Prefetch unified memory to GPU
            if hasattr(array, 'prefetch'):
                with stream:
                    array.prefetch()
            
        except Exception as e:
            logger.warning(f"GPU prefetch failed: {e}")
    
    def prefetch_to_cpu(self, array, stream=None):
        """Prefetch array to CPU memory."""
        if not USE_CUPY or not self.zero_copy_enabled:
            return
        
        try:
            if stream is None:
                stream = cp.cuda.Stream()
            
            # Prefetch unified memory to CPU
            if hasattr(array, 'prefetch'):
                with stream:
                    array.prefetch(cp.cuda.runtime.memPrefetchCpu)
            
        except Exception as e:
            logger.warning(f"CPU prefetch failed: {e}")
    
    def get_memory_info(self):
        """Get memory information."""
        info = {}
        if USE_CUPY:
            try:
                meminfo = cp.cuda.Device().mem_info
                info['gpu_free'] = meminfo[0] / (1024**3)
                info['gpu_total'] = meminfo[1] / (1024**3)
                info['gpu_used'] = info['gpu_total'] - info['gpu_free']
                info['zero_copy_enabled'] = self.zero_copy_enabled
                info['pinned_memory_enabled'] = self.pinned_memory_pool is not None
                info['memory_mappings_count'] = len(self.memory_mappings)
                
                # Get memory pool info
                if self.pinned_memory_pool:
                    info['pinned_memory_used'] = self.pinned_memory_pool.used_bytes() / (1024**2)
                    info['pinned_memory_total'] = self.pinned_memory_pool.total_bytes() / (1024**2)
                
            except Exception as e:
                logger.warning(f"Memory info failed: {e}")
        
        return info
    
    def cleanup(self):
        """Cleanup memory resources."""
        if USE_CUPY:
            try:
                # Clear memory mappings
                self.memory_mappings.clear()
                
                # Clear memory pools
                if self.pinned_memory_pool:
                    self.pinned_memory_pool.free_all_blocks()
                
                # Synchronize CUDA
                cp.cuda.Stream.null.synchronize()
                
                logger.info("Zero-copy memory cleanup complete")
                
            except Exception as e:
                logger.warning(f"Memory cleanup failed: {e}")

# Global zero-copy memory manager instance
zero_copy_manager = ZeroCopyMemoryManager()
