from aiogram import Router, F
from aiogram.fsm.state import State, StatesGroup

class BotStates(StatesGroup):
    price = State()
    choosing_price = State()
    uslugi = State()
    chatgpt = State()
    api_key = State()


router_chat = Router()
