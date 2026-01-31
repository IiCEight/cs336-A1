import sys
from loguru import logger

# Get rid of the default plain formatting
logger.remove() 

# Add a new format
# use level to filter some noisy messages
# logger.add(sys.stdout, 
#            colorize=True, 
#            level="WARNING",
#            format=  "{time: HH:mm:ss} " \
#                     "{level: <10} " \
#                     "{name: <8} " \
#                     "{function: <8} " \
#                     "{line: <3} "
#                     "<level>{message}</level>")

# ------------ filter --------
# 1. How the Filter Works
# The filter parameter in logger.add() accepts a function. 
# This function receives a Record (a dictionary containing 
# all the metadata about the log) and must return True 
# (to log it) or False (to drop it).

# The Record Dictionary
# To write a good filter, you need to know what’s inside the record. Here are the most common keys:
# record["name"]: The name of the module (e.g., __main__ or utils.db).
# record["level"].name: The level as a string (e.g., "DEBUG").
# record["message"]: The actual text of the log.
# record["extra"]: Any data you passed using .bind().

def customFilter(record):
    return "saber" in record["message"].lower()

logger.add(sys.stdout, 
           colorize=True, 
           level="WARNING",
        #    filter= lambda record: record["name"] == "__main__" ,
            filter = customFilter,
           format=  "{time: HH:mm:ss} " \
                    "{level: <10} " \
                    "{name: <8} " \
                    "{function: <8} " \
                    "{line: <3} "
                    "<level>{message}</level>")

# ----------- format -----------
# {time}	The current timestamp (supports custom formatting like YYYY-MM-DD)
# {level}	The severity level (INFO, DEBUG, etc.)
# {message}	The actual text you logged
# {name}	The module name (e.g., main)
# {function}	The function name where the log was called
# {line}	The line number
# {extra}	Any data passed via .bind()

# ----------- align -----------
# Inside your curly braces { }, you can specify a minimum 
# width and the alignment using the : syntax.

# {token: <10}: Left-aligned, minimum 10 characters.
# {token: >10}: Right-aligned, minimum 10 characters.
# {token: ^10}: Center-aligned, minimum 10 characters.


# NOTE: it will output to stderr!
logger.debug("Debugging details")
logger.info("General info")
logger.success("It worked!")
logger.warning("Something is fishy")
logger.error("Something went wrong")
logger.critical("The system Saber is down!")



