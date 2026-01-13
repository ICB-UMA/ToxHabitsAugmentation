#!/usr/bin/env python3
"""
Logging configuration for the application.
Provides structured logging with different log levels and output formats.
"""

# --- Third-Party Imports ---
from logging.handlers import RotatingFileHandler

# --- Local Imports ---
from ..imports import (
    sys,
    logging,
    datetime,
    json,
    Any,
    Dict,
    Optional,
    Path
)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_object = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_object.update(record.extra)
        
        return json.dumps(log_object)

class LogManager:
    """Manages logging configuration and setup."""
    
    def __init__(self, app_name: str = "MyApp"):
        """
        Initialize the log manager.
        
        Args:
            app_name: Name of the application for log naming
        """
        self.app_name = app_name
        self.loggers = {}
    
    def setup_logging(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        console_output: bool = True
    ) -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (if None, no file logging)
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
            console_output: Whether to output to console
            
        Returns:
            Configured root logger
        """
        # Convert string level to logging constant
        level = getattr(logging, log_level.upper())
        
        # Create logs directory if needed
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        json_formatter = JSONFormatter()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler (JSON format)
        if log_file:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(json_formatter)
            root_logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        root_logger.propagate = False
        
        return root_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a named logger.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured logger
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    @staticmethod
    def log_with_context(
        logger: logging.Logger,
        level: str,
        message: str,
        extra_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Log a message with additional context.
        
        Args:
            logger: Logger instance
            level: Log level
            message: Log message
            extra_context: Additional context as dictionary
            **kwargs: Additional key-value pairs for context
        """
        context = extra_context or {}
        context.update(kwargs)
        
        log_method = getattr(logger, level.lower())
        
        if context:
            # Create a new log record with extra fields
            extra = {'extra': context}
            log_method(message, extra=extra)
        else:
            log_method(message)


# Global log manager instance
log_manager = LogManager()

# Convenience functions
def setup_logging(**kwargs):
    """Quick setup for logging."""
    return log_manager.setup_logging(**kwargs)

def get_logger(name: str = __name__) -> logging.Logger:
    """Get a logger instance."""
    return log_manager.get_logger(name)

def log_error(error: Exception, logger: Optional[logging.Logger] = None, **context):
    """Convenience function for logging exceptions."""
    if logger is None:
        logger = get_logger(__name__)
    
    logger.error(
        f"Exception occurred: {str(error)}",
        exc_info=True,
        extra=context
    )

# Example usage decorator
def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"Function {func.__name__} executed in {execution_time:.4f} seconds",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time
                    }
                )
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {execution_time:.4f} seconds",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time,
                        'error': str(e)
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator

# -- Create a default logger instance --

# Setup logging
setup_logging(
    log_level="DEBUG",
    log_file="logs/app.log",
    console_output=True
)

# Default logger for module
logger = get_logger(__name__)
