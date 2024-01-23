from bot.bot import router_chat as router, BotStates
from aiogram import filters, types, F
from aiogram.fsm.context import FSMContext
from bot.bot_constants import bot_constants
from bot.gpt_bot import get_gpt_helper
from db.botdb import BotDB


@router.message(filters.Command("start"))
async def cmd_start(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    await message.answer( bot_constants.start, reply_markup=types.ReplyKeyboardRemove())
    await h.select_choice(BotStates.chatgpt, bot_constants.char_gpt)

@router.message(filters.Command("api_key"))
@router.message(filters.Command("api"))
async def cmd_api(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    await h.select_choice(BotStates.api_key, bot_constants.api_key)

@router.callback_query(lambda c: c.data.startswith('rate_'))
async def handle_rating(callback_query: types.CallbackQuery):
    print("rate", callback_query.data)
    data_parts = callback_query.data.split('_') # Разделяем данные в callback_data
    uid, rate = map(int,data_parts[1:])
    h = await get_gpt_helper(callback_query.message, None)
    await h.message.answer(text =f"оценка {rate}",reply_markup= types.ReplyKeyboardRemove())
    text = h.message.text
    # print(text)
    await h.message.edit_text(text = text, reply_markup=None)
    h.db.usermessage_id = uid
    await h.db.add_rating(rate)
    await h.close()

@router.message(filters.Command(commands=["clear"]))
@router.message(filters.Command(commands=["cancel"]))
@router.message(F.text.lower() == "отмена")
async def cmd_cancel(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    await h.answer("Действие отменено")
    h.data.clear()
    await h.update_data()

@router.message(filters.Command(commands=["dialog"]))
async def cmd_dialog(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    await h.answer("Включен режим диалога")
    h.data.dialog = 1
    await h.update_data()

@router.message(filters.Command(commands=["nodialog"]))
async def cmd_nodialog(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    await h.answer("Выключен режим диалога")
    h.data.dialog = 0
    await h.update_data()
    await h.close()


@router.message(filters.Command(commands=["status"]))
async def cmd_status(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    if h.api_key:
        text = "Введен api_key, неограниченное количество запросов"
    else:
        text = f"Осталось {h.data.quota} токенов,начало {h.data.dt_startquota}"
    await h.answer(text)
    await h.close()

@router.message(F.text.startswith("/"))
async def cmd_notsupport(message: types.Message, state:FSMContext):
    await message.answer("Команда не поддерживается")

@router.message(BotStates.api_key, F.text)
async def message_api_key(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    await h.save_api_key()
    await h.select_choice(BotStates.chatgpt, bot_constants.api_key_saved)
    await h.close()

@router.message(BotStates.chatgpt, F.text)
@router.message(F.text)
async def message_with_text(message: types.Message, state:FSMContext):
    h = await get_gpt_helper(message, state)
    await h.answer_gpt()
    await h.close()

