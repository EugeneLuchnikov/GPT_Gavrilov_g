from utils.initconf_env import config
from utils.logger import logger
logger.info("Start. Загружаем индексную БД")

from db.base import init_database
init_database(config.database_path)
logger.info("Загружаем БД")
from gptindex.ixapp import ixapp
from bot.start_bot import bot_main
import asyncio

ixapp.init()
from db.base import database

async def main():
    await database.async_main()
    await bot_main()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Exit')
