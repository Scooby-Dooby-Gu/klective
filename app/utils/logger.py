import logging
import sys

def setup_logger():
    """Set up and configure the logger."""
    # Create logger
    logger = logging.getLogger('klective')
    logger.setLevel(logging.DEBUG)

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger 