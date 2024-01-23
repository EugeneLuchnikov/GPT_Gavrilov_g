from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.types import (InlineKeyboardMarkup, InlineKeyboardButton)


def make_row_keyboard(items: list[str]) -> ReplyKeyboardMarkup:
    """
    Создаёт реплай-клавиатуру с кнопками в один ряд
    :param items: список текстов для кнопок
    :return: объект реплай-клавиатуры
    """
    row = [KeyboardButton(text=item) for item in items]
    return ReplyKeyboardMarkup(keyboard=[row], resize_keyboard=True, one_time_keyboard=True)

def make_keyboard(items: list[str]) -> ReplyKeyboardMarkup:
    """
    Создаёт реплай-клавиатуру с кнопками в один столбец
    :param items: список текстов для кнопок
    :return: объект реплай-клавиатуры
    """
    row = [[KeyboardButton(text=item)] for item in items]
    return ReplyKeyboardMarkup(keyboard=row, resize_keyboard=True)

def get_yes_no_kb() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardBuilder()
    kb.button(text="Да")
    kb.button(text="Нет")
    kb.adjust(2)
    return kb.as_markup(resize_keyboard=True)

# клавиатура для оценки ответов
def create_rating_keyboard(uid):
    # Создаем список кнопок для каждой оценки
    buttons = [
        InlineKeyboardButton(text=str(rate), callback_data=f'rate_{uid}_{rate}')
        for rate in [-2, -1, 0, 1, 2]
    ]
    # Создаем инлайн-клавиатуру с этими кнопками
    keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])
    return keyboard

# клавиатура для оценки ответов
def create_choice_keyboard(count, name="choice"):
    # Создаем список кнопок для каждой оценки
    buttons = [
        InlineKeyboardButton(text=str(rate), callback_data=f'{name}_{rate}')
        for rate in range(count)
    ]
    ikb =[]
    row= None
    for i, b in enumerate(buttons):
        if i%8==0:
            row=[]
            ikb.append(row)
        row.append(b)
    # Создаем инлайн-клавиатуру с этими кнопками
    keyboard = InlineKeyboardMarkup(inline_keyboard=ikb)
    return keyboard
