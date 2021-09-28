import logging

def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    return logging.getLogger(name)
