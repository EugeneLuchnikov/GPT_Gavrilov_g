from config import config
from aiogram import Bot, Dispatcher
from bot.commands import router as router_chat
from bot.different_types import router as router_diff
from utils.logger import logger

async def bot_main():
    dp = Dispatcher()
    bot = Bot(token=config.TG_TOKEN)

    dp.include_routers(router_chat, router_diff)
    logger.info('Бот успешно запущен!')
    await dp.start_polling(bot)
