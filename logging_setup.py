import queue
import logging
import json
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

# Load configuration from a JSON file
def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

# Set up logging with rotation
def setup_logging():
    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)
    log_file_handler = RotatingFileHandler('video_processing.log', maxBytes=5*1024*1024, backupCount=5)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(queue_handler)
    listener = QueueListener(log_queue, log_file_handler)
    listener.start()
    return logger, listener

# Usage in other parts
config = load_config()
logger, listener = setup_logging()
