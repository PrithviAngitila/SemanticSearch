import logging
import os
from datetime import datetime

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"[{datetime.utcnow().isoformat()}] [{record.levelname}] {record.msg}"
        return super().format(record)

def setup_logger(log_file_path="app.log"):
    log_level = logging.INFO

    # Create an absolute path for the log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    # If the specified path already exists and is a file, create a new directory with a different name
    if os.path.isfile(log_dir):
        log_dir = os.path.join(log_dir, "logs_" + datetime.now().strftime("%Y%m%d%H%M%S"))

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Setup logger
    logger = logging.getLogger("app_logger")
    logger.setLevel(log_level)

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_path))
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = CustomFormatter()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
