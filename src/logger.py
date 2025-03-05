import logging
import os

def setup_logger(log_file, log_level):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("AIMET_Pipeline")
    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
    
    logger.setLevel(log_levels.get(log_level, logging.INFO))

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
