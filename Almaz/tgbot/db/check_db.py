from utils.initconf_env import config
from utils.logger import logger
import asyncio

from db.base import init_database
logger.debug("init_database")

init_database(config.database_path)
from db.base import database
async def main():
    await database.async_main()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exit')

