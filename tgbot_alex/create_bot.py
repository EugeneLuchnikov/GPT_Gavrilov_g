from dotenv import load_dotenv
import os
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

load_dotenv(override=True)

TOKEN = str(os.getenv("TOKEN"))
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

GSERVICEACCOUNTFILE = str(os.getenv("GSERVICEACCOUNTFILE"))
SHEETID_PARAM = str(os.getenv("SHEETID_PARAM"))

POSTGRE_HOST=str(os.getenv('POSTGRE_HOST'))
POSTGRE_DB=str(os.getenv('POSTGRE_DB'))
POSTGRE_USER=str(os.getenv('POSTGRE_USER'))
POSTGRE_PASSW=str(os.getenv('POSTGRE_PASSW'))
POSTGRE_PORT=str(os.getenv('POSTGRE_PORT'))

storage = MemoryStorage()

bot = Bot(token=TOKEN, parse_mode=None)
dp = Dispatcher(storage=storage)


