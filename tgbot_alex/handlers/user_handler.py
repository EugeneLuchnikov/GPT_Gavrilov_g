import json
import asyncio
from aiogram import Router
from aiogram import types
from aiogram import F
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from datetime import datetime

from core.utils import set_users_into_gsh, save_data_to_google_sheets
from create_bot import bot
from core import main_chatgpt
from dbase.models import User, History
from dbase.repository import add_user, add_history, user_exists, get_user, get_user_id, \
    update_last_question_time, update_last_interaction, update_dialog_state, \
    update_dialog_state_and_score, update_dialog_statistics, get_dialog_state, get_num_queries, get_history_for_dialog
from keyboards.user_keyboard import drating_inline_buttons_keyboard
from bot import logger
from handlers.admin_handler import ADMIN_CHAT_ID
from config import NUMBER_FREE_QUESTIONS, UNLIM_QUESTION

router = Router()  # [2]

welcome_message = "<b>Добро пожаловать!</b> 🙌🏻 \n\nЯ - полезный помощник, на основе ChatGPT.\nНейроконсультант " \
                  "Германа Гаврилова.\nЯ работаю в демонстрационном режиме " \
                  f"поэтому у вас есть {NUMBER_FREE_QUESTIONS} запросов ко мне.\nСоветую прочитать рекомендации " \
                  "прежде чем начать общаться со мной ➡️ /recommendations.\n\n " \
                  "Мы подготовили несколько примеров вопросов, которые можно задать боту (просто нажми на команду с " \
                  "нужным вопросом):\n" \
                  "/Example1(Привет! Расскажи про ... Что это?)\n" \
                  "/Example2(Что такое ..?)\n" \
                  "/Example3(Как ...?)"


@router.message(Command(commands=["start"]))
async def cmd_start(message: types.Message):
    if not await user_exists(message.from_user.id):
        user = User(
            tg_id = message.from_user.id,
            e_mail = None,
            first_name = message.from_user.first_name,
            last_name = message.from_user.last_name,
            username = message.from_user.username,
            last_interaction = datetime.utcnow(),
            last_dialog = None,
            last_question = None,
            last_answer = None,
            last_chunks = None,
            last_num_token = None,
            last_cost = None,
            dialog_state = "finish",
            dialog_score = None,
            last_question_time = None,
            last_time_duration = None,
            num_queries = 0
        )
        await add_user(user)
        try:
            set_users_into_gsh()
            logger.info(f"Добавили нового user в Google-таблицу!")
        except Exception as error:
            logger.warning(
                f"Ошибка добавления записи Пользователи: {error}")
        await message.reply(welcome_message + "\n\nЗадайте свой вопрос...", parse_mode='HTML')
        await update_dialog_state(message.from_user.id, 'start')
    else:
        if await get_dialog_state(message.from_user.id) == 'close':
            await bot.send_message(message.from_user.id, "Оцените предыдущий ответ чтобы продолжить использование "
                                                         "помощника.")
        else:
            await message.reply(welcome_message + "\n\nЗадайте свой вопрос...", parse_mode='HTML')
            await update_dialog_state(message.from_user.id, 'start')
    dialog_status = await get_dialog_state(message.from_user.id)
    #print(f'user_handler: cmd_start: {dialog_status = }')


@router.message(lambda message: asyncio.run(get_dialog_state(message.from_user.id)) == 'close')
async def any_action(message: types.Message):
    await bot.send_message(message.from_user.id, "Оцените предыдущий ответ чтобы продолжить использование помощника.")


@router.message(Command(commands=['recommendations']))
async def send_recommendations(message: types.Message):
    """
    This handler will be called when user sends `/recommendations` command
    """
    recommendations_student = '''
    Я - полезный помощник Германа Гаврилова основе ChatGPT.
    Советую прочитать полностью прежде чем начать общаться со мной.

    1. Сложные вопросы я рекомендую задавать мне на английском языке, так же на английском языке я работаю быстрее.

    2. Будьте конкретны и ясны: задавая вопрос, предоставьте как можно больше релевантной информации, чтобы помочь 
    мне лучше понять, что вам нужно. 

    3. Используйте правильную грамматику и орфографию: так мне будет легче понять ваше сообщение и ответить на него.

    4. Будьте вежливы и корректны в высказываниях: использование вежливых выражений и проявление уважения могут иметь 
    большое значение для обеспечения позитивного взаимодействия. 

    5. Избегайте использования аббревиатур или текстовой речи: это может затруднить мне понимание того, 
    что вы пытаетесь донести. 

    6. Задавайте по одному вопросу за раз: Если у вас есть несколько вопросов, лучше задавать их по одному, 
    чтобы помочь мне дать четкий и целенаправленный ответ. 

    7. Укажите контекст: Если ваш вопрос связан с определенной темой, предоставьте некоторую справочную информацию, 
    которая поможет мне понять, о чем вы спрашиваете. 

    8. Избегайте использования всех заглавных букв: ввод текста всеми заглавными буквами часто воспринимается как 
    крик и может затруднить продуктивную беседу. 

    9. Будьте терпеливы: я являюсь языковой моделью искусственного интеллекта, и мне может понадобиться некоторое 
    время, чтобы обработать ваш запрос. Ответы на некоторые вопросы могут длиться до 2 минут. 

    Готов общаться в чате. 
    И скажите Герману, что я уже работаю.
    '''

    await message.reply(recommendations_student, parse_mode='HTML')


@router.message(Command(commands=['Example1']))
async def send_question_example1(message: types.Message):
    """
    This handler will be called when user sends `/recommendations` command
    """
    question_example1 = '''
    <b>Вопрос1</b>
    
    Привет!
    Расскажи про ... Что это?
    
    <b>Ответ ChatGPT:</b> 
    Привет! Тут будет ответ 1. 
    '''

    await message.reply(question_example1, parse_mode='HTML')


@router.message(Command(commands=['Example2']))
async def send_question_example2(message: types.Message):
    """
    This handler will be called when user sends `/recommendations` command
    """
    question_example2 = '''
    <b>Вопрос2</b>

    Что такое ...?

    <b>Ответ ChatGPT:</b> 
    ... - это основной 
     Тут будет ответ 2'''

    await message.reply(question_example2, parse_mode='HTML')


@router.message(Command(commands=['Example3']))
async def send_question_example1(message: types.Message):
    """
    This handler will be called when user sends `/recommendations` command
    """
    question_example3 = '''
    <b>Вопрос3</b>

    Как ... ?
    
    <b>Ответ ChatGPT:</b> Тут будет ответ 3'''

    await message.reply(question_example3, parse_mode='HTML')


@router.message(Command(commands=['balance']))
async def send_balance(message: types.Message):
    pass


@router.message(Command(commands=['context']))
async def reset_context(message: types.Message):
    pass


@router.callback_query(lambda c: c.data.startswith("drate_"))
async def process_callback_qrating(callback_query: types.CallbackQuery):
    if await get_dialog_state(callback_query.from_user.id) == 'close':
        user = await get_user(callback_query.from_user.id)   # получим из БД информацию о пользователе
        #print(f'process_callback_qrating: {user_data = }')
        # print(f'process_callback_qrating: {score_chuncks = }')
        rating = int(callback_query.data[6:])
        #print(f'process_callback_qrating: {type(rating)}, {rating = }')
        #await bot.answer_callback_query(callback_query.id, text=f"Спасибо за вашу оценку: {rating}!", show_alert=True)
        if UNLIM_QUESTION or callback_query.from_user.id in ADMIN_CHAT_ID:
            await bot.send_message(callback_query.from_user.id, f"Спасибо за вашу оценку: {rating}! Можете задать "
                                                                f"следующий вопрос")
        else:
            await bot.send_message(callback_query.from_user.id, f"Спасибо за вашу оценку: {rating}! Можете задать "
                                                                f"следующий вопрос (осталось "
                                                                f"{int(NUMBER_FREE_QUESTIONS - (await get_num_queries(callback_query.from_user.id)))} запрос(ов).")
        # Здесь сохраняется оценка пользователя для дальнейшего анализа или использования
        await update_dialog_state_and_score(callback_query.from_user.id, 'finish', rating)

        # переда записью истории проверим содержимое user_data:
        # for i, item in enumerate(user_data):
        #     print(f'User_data[{i}]. {item}')

        user_id = await get_user_id(callback_query.from_user.id)
        # Запись истории
        history = History(
            user_id = user_id,
            question = user.last_question,
            answer = user.last_answer,
            score_name = "question",
            score_text = "\n".join([f'Пользователь: {user.last_question}', f'Ассистент: {user.last_answer}']),
            score_chunk = user.last_chunks,
            score = rating,
            num_token = user.last_num_token,
            cost = user.last_cost,
            question_time = user.last_question_time,
            time_duration = user.last_time_duration,
            score_time = datetime.utcnow()
        )

        # переда записью истории проверим содержимое history_data:
        # for i, item in enumerate(history_data):
        #     print(f'history_data[{i}]. {item}')

        await add_history(history)
        try:
            logger.info(
                f"Оценка вопроса! Добаление записи в таблицу Оценка")
            save_data_to_google_sheets('Оценки', history)
        except Exception as error:
            logger.warning(
                f"Ошибка добавления записи Оценки: {error}")
    await bot.answer_callback_query(callback_query.id)


@router.message(lambda message: asyncio.run(get_dialog_state(message.from_user.id)) in ['start', 'finish'])
async def generate_answer(message: types.Message):
    #print(f'generate_answer: starting...')
    await update_last_interaction(message.from_user.id, datetime.utcnow())
    num_queries = await get_num_queries(message.from_user.id)
    print(f'generate_answer: {UNLIM_QUESTION = } {num_queries = }')
    if UNLIM_QUESTION or num_queries < NUMBER_FREE_QUESTIONS or message.from_user.id in ADMIN_CHAT_ID:       # ограничение по ответам: менее NUMBER_FREE_QUESTIONS ответов или админ - неограничено
        try:
            msg = await message.answer("Идет подготовка ответа. Ждите...⏳")  # msg["message_id"]
            user_id = await get_user_id(message.from_user.id)
            history_items = await get_history_for_dialog(user_id)
            time1 = datetime.utcnow()
            await update_last_question_time(message.from_user.id, time1)
            logger.info(f"Запрос пошел: {message.text}")
            completion, dialog, chunks, cost_request = await main_chatgpt.WorkerOpenAI().get_chatgpt_answer(message.text, history_items)
            logger.info(f"Запрос вернулся: [completion]")
            time2 = datetime.utcnow()
            duration = time2 - time1
            await msg.edit_text(completion.choices[0].message.content)
            
            last_chunks = '\n '.join([f'\n==  ' + doc.page_content + '\n' for doc in chunks])

            await update_dialog_statistics(
                message.from_user.id, json.dumps(dialog), message.text, completion.choices[0].message.content,
                last_chunks, completion.usage.total_tokens, cost_request, 'close', duration.total_seconds(), num_queries + 1
            )

            await message.answer("Пожалуйста, оцените качество консультации от -2 до 2:",
                                 reply_markup=drating_inline_buttons_keyboard())
        except Exception as error:
            logger.warning(f"Ошибка генерации: {error}")
            logger.exception(error)
            await bot.send_message(message.from_user.id, f"ОШИБКА: {error}")
            await bot.send_message(message.from_user.id, "Модель в настоящее время перегружена. Попробуйте позже.")
    else:
        await bot.send_message(message.from_user.id, f"Вы исчерпали всё количество запросов ({NUMBER_FREE_QUESTIONS}) демонстрационного "
                                                     "режима.\nСпасибо что воспользовались нашим Помощником! 🤝")


async def generate_algorithm_error(message: types.Message):
    logger.warning(f"Ошибка алгоритма бота. Сообщение пользователя не обработано")
    await message.answer("Извините, сбой в алгоритме Бота: ваше сообщение не обработано")
