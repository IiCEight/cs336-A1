from loguru import logger
import sys

def set_up_logger(level:str):

    logger.remove()

    logger.add(sys.stdout, 
            # colorize=True, 
            level=   level,
            format=     "<green>{time:HH:mm:ss}</green> | " \
                        "<level>{level: <6}</level> | " \
                        "{name: <8} | " \
                        "{function: <8} | " \
                        "{line: <3} | " \
                        "<level>{message}</level>")