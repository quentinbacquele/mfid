import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime

class MFIDLogger:
    """
    Configurable logging system for the MFID application
    
    Features:
    - Multiple logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - File and console output
    - Rotating file handler to manage log size
    - Custom log formatting
    
    Example usage:
    ```
    from mfid.utils.logging_utils import get_logger
    
    logger = get_logger('face_detection')
    logger.info('Starting face detection')
    try:
        # some code that might fail
        pass
    except Exception as e:
        logger.error(f'Error in face detection: {e}', exc_info=True)
    ```
    """
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize the logging system"""
        if cls._initialized:
            return
        
        # Create logs directory in user's home directory
        log_dir = os.path.join(str(Path.home()), '.mfid', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up the root logger
        root_logger = logging.getLogger('mfid')
        root_logger.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler for all logs
        all_logs_file = os.path.join(log_dir, 'mfid.log')
        file_handler = RotatingFileHandler(
            all_logs_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create error file handler
        error_logs_file = os.path.join(log_dir, 'errors.log')
        error_file_handler = RotatingFileHandler(
            error_logs_file,
            maxBytes=2*1024*1024,  # 2MB
            backupCount=3
        )
        error_file_handler.setLevel(logging.ERROR)
        
        # Create formatters and add them to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        error_file_handler.setFormatter(formatter)
        
        # Add handlers to the root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name):
        """
        Get a logger for a specific module or component
        
        Args:
            name (str): Name of the module or component
            
        Returns:
            logging.Logger: Logger instance
        """
        cls.initialize()
        
        logger_name = f'mfid.{name}'
        
        if logger_name in cls._loggers:
            return cls._loggers[logger_name]
        
        logger = logging.getLogger(logger_name)
        cls._loggers[logger_name] = logger
        
        return logger
    
    @classmethod
    def log_exception(cls, logger, message, exc_info=None):
        """
        Log an exception with appropriate context
        
        Args:
            logger (logging.Logger): Logger instance
            message (str): Error message
            exc_info: Exception information (optional)
        """
        logger.error(message, exc_info=exc_info if exc_info else True)
    
    @classmethod
    def get_logs_location(cls):
        """Get the location of the log files"""
        return os.path.join(str(Path.home()), '.mfid', 'logs')
    
    @classmethod
    def create_session_log(cls):
        """Create a new session log file with timestamp"""
        cls.initialize()
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(str(Path.home()), '.mfid', 'logs', 'sessions')
        os.makedirs(log_dir, exist_ok=True)
        
        session_log_file = os.path.join(log_dir, f'session_{timestamp}.log')
        handler = logging.FileHandler(session_log_file)
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger('mfid')
        root_logger.addHandler(handler)
        
        return session_log_file, handler
    
    @classmethod
    def end_session_log(cls, handler):
        """Close a session log handler"""
        if handler:
            handler.close()
            root_logger = logging.getLogger('mfid')
            root_logger.removeHandler(handler)

# Convenience function to get a logger
def get_logger(name):
    """Get a logger for a specific module"""
    return MFIDLogger.get_logger(name)

# Convenience function to log exceptions
def log_exception(logger, message, exc_info=None):
    """Log an exception with context"""
    MFIDLogger.log_exception(logger, message, exc_info) 