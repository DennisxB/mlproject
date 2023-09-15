import os
import logging
from datetime import datetime

# Define log filename and setup logs path
LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")       
os.makedirs(logs_path, exist_ok=True)                                   # make log path directory

# Create log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# process logging
logger = logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("logger has started")