import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Regex patterns for redaction
API_KEY_PATTERN = re.compile(r'(?i)(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\',}\s]+)')
SK_PATTERN = re.compile(r'sk-[a-zA-Z0-9]{20,}')

class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive API keys from log messages."""
    
    def filter(self, record):
        msg = record.getMessage()
        if isinstance(msg, str):
            msg = API_KEY_PATTERN.sub(r'\g<1>***REDACTED***', msg)
            msg = SK_PATTERN.sub('***REDACTED***', msg)
            record.msg = msg
            record.args = () # args already interpolated
        return True

def setup_logger(name: str = "ytclipper", log_dir: str | Path | None = None) -> logging.Logger:
    """Setup and return a configured logger instance."""
    logger = logging.getLogger(name)
    
    # If already configured (handlers > 0), return it
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    redact_filter = SensitiveDataFilter()
    
    # Resolve log directory
    if log_dir is None:
        # Default to ./output in project root or current dir
        log_dir = Path.cwd()
    
    log_path = Path(log_dir) / "app.log"
    
    # Ensure directory exists immediately before creating handler
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        file_handler = RotatingFileHandler(
            str(log_path), 
            maxBytes=10*1024*1024, 
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redact_filter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if file system is read-only or path is invalid
        print(f"Failed to setup file logging at {log_path}: {e}")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Console is less verbose
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redact_filter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Global logger instance exported
logger = setup_logger()

__all__ = ['logger', 'setup_logger']
