from utils.initconf_env import config, ppo
from db.base import init_database
from utils.logger import logger
logger.debug("start")
# from gptindex.ixapp import ixapp
# ixapp.init()

# from gptindex.ixapp_test import test
# test()

init_database(config.database_path)


# ppo(config)
# @logger.catch
# def ee():
#     return 1/0
# from bot.bot_constants import bot_constants
# print(bot_constants.start)

from db.db_test import Test
t=Test()
t.user_tg_id+=3+10+39+5
import asyncio
async def main():
    await t.test1()
#     t.user_tg_id +=1
#     await t.test2()
#     print(t.user_tg_id, t.user_id)
#     await  t.test3()
#
asyncio.run(main())
# ee()
logger.debug("end")
