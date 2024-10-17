import logging
import os
import sys

def __setup_logger():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    # Use the current working directory for the log file
    log_file_path = os.path.join(os.getcwd(), "scraper-indexer", "app.log")
    file_handler = logging.FileHandler(log_file_path)

    # Create formatters and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

logger = __setup_logger()
