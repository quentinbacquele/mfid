import os
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtCore import QObject, pyqtSignal, QMutex

from mfid.utils.logging_utils import get_logger

logger = get_logger('batch')

class BatchProcessor(QObject):
    """
    Handles batch processing of files with progress tracking and parallel execution
    
    Features:
    - Queue management for processing multiple files
    - Progress tracking and reporting
    - Parallel processing with thread pool
    - Error handling and reporting
    
    Example usage:
    ```python
    def process_file(file_path, **kwargs):
        # Process a single file
        return {'result': 'Success', 'file_path': file_path}
    
    # Create processor with 4 worker threads
    processor = BatchProcessor(worker_count=4)
    
    # Set up progress callback
    def progress_callback(current, total, result=None):
        print(f"Progress: {current}/{total}")
        if result:
            print(f"Processed: {result['file_path']}")
    
    # Add files to process
    for file_path in file_list:
        processor.add_task(process_file, file_path)
    
    # Start processing with progress callback
    processor.process(progress_callback)
    
    # Get results
    results = processor.get_results()
    ```
    """
    
    # Signals for progress updates
    progress_updated = pyqtSignal(int, int, object)  # current, total, result
    processing_started = pyqtSignal()
    processing_completed = pyqtSignal()
    processing_error = pyqtSignal(str, str)  # error message, file path
    
    def __init__(self, worker_count=None):
        """
        Initialize the batch processor
        
        Args:
            worker_count (int, optional): Number of worker threads. Defaults to CPU count.
        """
        super().__init__()
        self.task_queue = queue.Queue()
        self.results = []
        self.results_lock = QMutex()
        self.is_processing = False
        self.stop_requested = False
        
        # Determine worker count (default to CPU count)
        self.worker_count = worker_count or min(os.cpu_count(), 8)
        logger.info(f"Batch processor initialized with {self.worker_count} workers")
    
    def add_task(self, task_func, *args, **kwargs):
        """
        Add a task to the processing queue
        
        Args:
            task_func (callable): Function to process a single file
            *args: Positional arguments for the task function
            **kwargs: Keyword arguments for the task function
        """
        self.task_queue.put((task_func, args, kwargs))
        logger.debug(f"Task added to queue. Queue size: {self.task_queue.qsize()}")
    
    def clear_tasks(self):
        """Clear all pending tasks from the queue"""
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break
        logger.info("Task queue cleared")
    
    def process(self, progress_callback=None):
        """
        Process all tasks in the queue
        
        Args:
            progress_callback (callable, optional): Callback for progress updates.
                Function signature: callback(current, total, result=None)
        """
        if self.is_processing:
            logger.warning("Batch processor is already running")
            return False
        
        self.is_processing = True
        self.stop_requested = False
        self.results = []
        
        # Get total task count
        total_tasks = self.task_queue.qsize()
        if total_tasks == 0:
            logger.warning("No tasks in queue to process")
            self.is_processing = False
            return False
        
        logger.info(f"Starting batch processing of {total_tasks} tasks")
        self.processing_started.emit()
        
        # Start a thread for processing
        threading.Thread(
            target=self._process_tasks,
            args=(total_tasks, progress_callback),
            daemon=True
        ).start()
        
        return True
    
    def _process_tasks(self, total_tasks, progress_callback=None):
        """
        Internal method to process tasks using a thread pool
        
        Args:
            total_tasks (int): Total number of tasks to process
            progress_callback (callable, optional): Callback for progress updates
        """
        completed_tasks = 0
        error_count = 0
        
        try:
            with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                # Submit all tasks
                future_to_task = {}
                while not self.task_queue.empty() and not self.stop_requested:
                    try:
                        task_func, args, kwargs = self.task_queue.get_nowait()
                        future = executor.submit(task_func, *args, **kwargs)
                        future_to_task[future] = (task_func.__name__, args)
                        self.task_queue.task_done()
                    except queue.Empty:
                        break
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    if self.stop_requested:
                        executor.shutdown(wait=False)
                        break
                    
                    task_name, args = future_to_task[future]
                    try:
                        result = future.result()
                        
                        # Store result
                        self.results_lock.lock()
                        self.results.append(result)
                        self.results_lock.unlock()
                        
                        # Update progress
                        completed_tasks += 1
                        self._update_progress(completed_tasks, total_tasks, result, progress_callback)
                        
                        logger.debug(f"Task '{task_name}' completed successfully")
                    except Exception as e:
                        error_count += 1
                        completed_tasks += 1
                        
                        # Get file path from args if available
                        file_path = args[0] if args else "unknown"
                        
                        error_msg = f"Error processing task '{task_name}': {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        self.processing_error.emit(error_msg, str(file_path))
                        
                        # Update progress even for failed tasks
                        self._update_progress(completed_tasks, total_tasks, None, progress_callback)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        finally:
            self.is_processing = False
            logger.info(f"Batch processing completed. {completed_tasks} tasks processed with {error_count} errors.")
            self.processing_completed.emit()
    
    def _update_progress(self, current, total, result, callback=None):
        """
        Update progress through signal and callback
        
        Args:
            current (int): Current number of completed tasks
            total (int): Total number of tasks
            result (object): Result of the completed task
            callback (callable, optional): Progress callback function
        """
        # Emit signal
        self.progress_updated.emit(current, total, result)
        
        # Call callback if provided
        if callback:
            try:
                callback(current, total, result)
            except Exception as e:
                logger.error(f"Error in progress callback: {str(e)}", exc_info=True)
    
    def stop(self):
        """Stop the batch processing"""
        if not self.is_processing:
            return
        
        logger.info("Stopping batch processing")
        self.stop_requested = True
    
    def get_results(self):
        """
        Get the results of all processed tasks
        
        Returns:
            list: List of results from all successfully completed tasks
        """
        self.results_lock.lock()
        results_copy = self.results.copy()
        self.results_lock.unlock()
        return results_copy
    
    def is_busy(self):
        """
        Check if the processor is currently processing tasks
        
        Returns:
            bool: True if processing, False otherwise
        """
        return self.is_processing


class BatchProcessingTask:
    """
    Base class for batch processing tasks
    
    This class can be extended to create specific task types
    with custom pre-processing and post-processing logic.
    
    Example:
    ```python
    class VideoProcessingTask(BatchProcessingTask):
        def process(self, video_path, **kwargs):
            # Process video
            return {'result': 'Success', 'file_path': video_path}
    
    processor = BatchProcessor()
    task = VideoProcessingTask()
    processor.add_task(task.process, 'path/to/video.mp4')
    ```
    """
    
    def __init__(self, name=None):
        """
        Initialize the task
        
        Args:
            name (str, optional): Task name for logging
        """
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f'batch.{self.name.lower()}')
    
    def process(self, *args, **kwargs):
        """
        Process a single item
        
        This method should be implemented by subclasses
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            object: Processing result
        """
        raise NotImplementedError("Subclasses must implement process() method") 