from loguru import logger
import sys

logger.remove()

logger.add(sys.stdout, 
           colorize=True, 
           level=   "DEBUG",
           format=  "{time: HH:mm:ss} " \
                    "{level: <10} " \
                    "{name: <8} " \
                    "{function: <8} " \
                    "{line: <3} "
                    "<level>{message}</level>")