import logging, sys

def setup_logger():
    import os
    logger = logging.getLogger("graph")
    if not logger.handlers:
        # Use DEBUG level if DEBUG env var is set
        level = logging.DEBUG if os.getenv("DEBUG") else logging.INFO
        logger.setLevel(level)
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

logger = setup_logger()
