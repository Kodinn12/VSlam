"""Thread manager with priority-based scheduling for SLAM system."""

import threading
import queue
import time
from enum import Enum
from typing import Callable, Optional, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ThreadPriority(Enum):
    """Thread priority levels."""
    CRITICAL = 0  # Tracking, frame processing
    HIGH = 1      # Mapping, integration
    MEDIUM = 2    # Optimization (BA, PGO)
    LOW = 3       # Loop closure, visualization


class ThreadTask:
    """Task wrapper for thread execution."""
    
    def __init__(self, func: Callable, args: tuple = (), kwargs: dict = None,
                 priority: ThreadPriority = ThreadPriority.MEDIUM):
        """
        Initialize task.
        
        Parameters
        ----------
        func : Callable
            Function to execute
        args : tuple
            Positional arguments
        kwargs : dict
            Keyword arguments
        priority : ThreadPriority
            Task priority
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = priority
        self.created_at = time.time()
    
    def __lt__(self, other):
        """Compare tasks for priority queue (lower priority value = higher priority)."""
        return self.priority.value < other.priority.value


class ThreadManager:
    """Manages thread pool with priority-based task scheduling."""
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize thread manager.
        
        Parameters
        ----------
        num_workers : int
            Number of worker threads
        """
        self.num_workers = num_workers
        self.task_queues: Dict[ThreadPriority, queue.Queue] = {
            priority: queue.Queue() for priority in ThreadPriority
        }
        self.workers = []
        self.shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
        logger.info(f"ThreadManager initialized with {num_workers} workers")
    
    def start(self):
        """Start all worker threads."""
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, daemon=True, name=f"Worker-{i}")
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {len(self.workers)} worker threads")
    
    def _worker(self):
        """Worker thread that processes tasks from priority queues."""
        while not self.shutdown_event.is_set():
            # Check queues in priority order (CRITICAL first)
            task = None
            for priority in sorted(ThreadPriority, key=lambda p: p.value):
                try:
                    task = self.task_queues[priority].get_nowait()
                    break
                except queue.Empty:
                    continue
            
            if task is None:
                # No tasks, sleep briefly
                time.sleep(0.001)
                continue
            
            try:
                task.func(*task.args, **task.kwargs)
            except Exception as e:
                logger.error(f"Task failed: {e}")
            finally:
                self.task_queues[task.priority].task_done()
    
    def submit(self, func: Callable, args: tuple = (), kwargs: dict = None,
              priority: ThreadPriority = ThreadPriority.MEDIUM) -> bool:
        """
        Submit task to thread manager.
        
        Parameters
        ----------
        func : Callable
            Function to execute
        args : tuple
            Positional arguments
        kwargs : dict
            Keyword arguments
        priority : ThreadPriority
            Task priority
        
        Returns
        -------
        bool
            True if task was submitted successfully
        """
        if self.shutdown_event.is_set():
            logger.warning("Thread manager is shutting down, cannot submit task")
            return False
        
        task = ThreadTask(func, args, kwargs, priority)
        self.task_queues[priority].put(task)
        return True
    
    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown thread manager gracefully.
        
        Parameters
        ----------
        timeout : float
            Timeout for worker threads to finish
        """
        self.shutdown_event.set()
        
        # Wait for queues to empty
        for priority_queue in self.task_queues.values():
            priority_queue.join()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        logger.info("Thread manager shut down")
    
    def get_queue_sizes(self) -> Dict[ThreadPriority, int]:
        """
        Get current queue sizes for each priority.
        
        Returns
        -------
        dict
            Queue sizes by priority
        """
        return {priority: q.qsize() for priority, q in self.task_queues.items()}
