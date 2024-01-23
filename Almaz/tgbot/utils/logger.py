from loguru import logger
from config import config
import os
import logging


log_path = os.path.join(config.index_path,"Logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)

logger.add(f"{log_path}/bot_debug.log", format="#{time} {level} {name} {message}", level="DEBUG", rotation="6 MB", compression="zip")
logger.add(f"{log_path}/bot_info.log", format="#{time} {level} {name} {message}", level="INFO", rotation="6 MB", compression="zip")

# Configure the logging system to write to a file
logging.basicConfig(
    filename=f'{log_path}/sqlalchemy.log',
    level=logging.INFO,
    format="#%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    encoding="utf-8"
    )


# using:
@logger.catch
def ee():
    return 1/0

def a1():
    try:
        1/0
    except ZeroDivisionError:
        logger.exception("What?!")
