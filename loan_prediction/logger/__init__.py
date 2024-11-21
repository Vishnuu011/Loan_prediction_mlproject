from math import log
import logging
import os
from datetime import datetime
from from_root import from_root

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir ="logs"
log_file_path = os.path.join(from_root(),log_dir,LOG_FILE)
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)