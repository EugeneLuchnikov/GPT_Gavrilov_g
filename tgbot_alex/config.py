"""
Настройки параметров и переменных проекта для разработчика.
Файл будет храниться во внутренней папке Docker,
доступа извне не будет в отличие от папки Settings.

В папке Settings должна быть проброшена из Docker в систему,
чтобы хранящиеся там переменные, можно было обновлять налету.

Так же и папка FAISS_DB_DIR должна быть проброшена из Docker в систему,
чтобы хранящиеся там индексы, можно было добавлять боту налету.
"""
import os
from core.gpt_models import GptModel
from enum import Enum

# OpenAI Models
cheap_model = GptModel(name='gpt-3.5-turbo-1106',
                       input_price=0.001, 
                       output_price=0.002, 
                       max_tokens_limit=16385, 
                       max_tokens_for_answer=2048
                       )

exp_model = GptModel(name='gpt-4-1106-preview', 
                     input_price=0.01, 
                     output_price=0.03,
                     max_tokens_limit=128000,
                     max_tokens_for_answer=2048
                     )

# Используемая модель
MODEL = cheap_model

# Максимальная длительность диалога в часах
MAX_DIALOG_PERIOD_IN_HOURS = 24
# Температура 
TEMPERATURE = 0.0
# Количество чанков
NUMBER_OF_CHUNKS = 8

# Максимальная длина сообщения в Telegram
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Промпты
DIALOG1_SYSTEM_FILE = 'dialog1_system.txt'
DIALOG1_USER_FILE = 'dialog1_user.txt'
DIALOG2_SYSTEM_FILE = 'dialog2_system.txt'
DIALOG2_USER_FILE = 'dialog2_user.txt'
DIALOG3_SYSTEM_FILE = 'dialog3_system.txt'
DIALOG3_USER_FILE = 'dialog3_user.txt'
DIALOG3_SUM_SYSTEM_FILE = 'dialog3_sum_system.txt'
DIALOG3_SUM_USER_FILE = 'dialog3_sum_user.txt'

# папки и пути
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))   # Корневой каталог
LOG_PATH = 'logs'               # хранение логов
FAISS_DB_DIR = 'faiss_indexes'  # хранение индексов
TXT_DB_DIR = 'txt_docs'
SETTINGS_PATH = 'settings'      # хранение внешних настроек, промптов

# Настройки логирования:
LOGGING_SERVICE = "gavrilov_bot"

# путь до внешней папки с настройками уведомлений TG_bot
APPRISE_CONFIG_PATH = "settings/apprise.yml"

# Настройка БД
RECREATE_DB = False  # удаляем ли старые таблицы при запуске бота (обновляем структуры таблицы, но теряем данные)
DB_TYPE = 'POSTGRE'
#DB_TYPE = 'SQLite3'

# Варианты поддержки диалога
class DialogAlgorithm(Enum):
    SEPARATE_MESSAGES = 1 # все реплики передаются в отдельные сообщения user/assistant
    SINGLE_MESSAGE = 2 # все реплики передаются в одно сообщение user
    SUMMARIZATION = 3 # все реплики саммаризируются перед подачей в user

# Текущий алгоритм поддержки диалога
DIALOG_ALGORITHM = DialogAlgorithm.SINGLE_MESSAGE

# Проверка бесплатных вопросов пользователя
NUMBER_FREE_QUESTIONS = 10
UNLIM_QUESTION = True