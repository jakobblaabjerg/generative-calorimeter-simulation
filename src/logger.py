import os
import logging

def setup_logger(name, save_dir):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_path = os.path.join(save_dir, f"{name}.log")

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Logger():

    def __init__(self, log_dir):
        
        self.log_dir = log_dir
        self.version = self._get_version()
        self.run_dir = os.path.join(log_dir, f"version_{self.version}")
        os.makedirs(self.run_dir, exist_ok=True)

    def _get_version(self):
        version = 0        
        while os.path.exists(os.path.join(self.log_dir, f"version_{version}")):
            version += 1
        return str(version) 

    def get_run_dir(self):
        return self.run_dir

    def log_metrics(self):
        pass

# TO DO:
# log metrics not implemented