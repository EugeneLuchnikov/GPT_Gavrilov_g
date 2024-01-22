import asyncio
import os
import tiktoken
from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import pandas as pd
from core.gpt_models import GptModel
from dbase.repository import get_df_users, get_df_history
from datetime import datetime
import gspread
import pygsheets
from create_bot import GSERVICEACCOUNTFILE, SHEETID_PARAM
from logger.logger import logger
from config import LOGGING_SERVICE, ROOT_DIR, SETTINGS_PATH

# Google Sheets setup
google_drive = None


# Метакласс для создания классов с единственным экземпляром в приложении
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def load_setting_file(file_name: str) -> str:
    setting_file = os.path.join(ROOT_DIR, SETTINGS_PATH, file_name)
    file_content = ""
    try:
        with open(setting_file, 'r', encoding='utf-8') as file:
            file_content = file.read()
            logger.info(f'Прочитали {file_name}')
    except Exception as e:
        print(f'Ошибка чтения {file_name}: {e}')
    return file_content


# пример подсчета токенов
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages: List[ChatCompletionMessageParam], model: str) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-1106",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-1106.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logger.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def creat_new_google_sheets(name):
    try:
        gc = pygsheets.authorize(service_file=GSERVICEACCOUNTFILE)
        sh = gc.create(name)
        sh.add_worksheet('Пользователи')
        sh.add_worksheet('Оценки')
    except Exception as e:
        print(f'creat_new_google_sheets: {e} ')
    return sh.url


def share_google_sheets(email_address):
    try:
        gc = pygsheets.authorize(service_file=GSERVICEACCOUNTFILE)
        sh = gc.open_by_url(f'https://docs.google.com/spreadsheets/d/{SHEETID_PARAM}')
        sh.share(email_address, role='writer', type='anyone')
        logger.info(f'share_google_sheets: Уcпешно!')
    except Exception as e:
        logger.error(f'share_google_sheets: {e}')


# TODO: Нужно переделать новый параметр history
def save_data_to_google_sheets(sheet_name, history):
    #print(f'save_data_to_google_sheets: Полный путь к файлу:{os.path.abspath(GSERVICEACCOUNTFILE)}')
    # if os.path.isfile(GSERVICEACCOUNTFILE):
    #     print(f"File exists {os.path.abspath(GSERVICEACCOUNTFILE)}")
    # else:
    #     print(f"File does not exist {os.path.abspath(GSERVICEACCOUNTFILE)}")

    try:
        gc = gspread.service_account(filename=GSERVICEACCOUNTFILE)  # Авторизуемся в Google Sheets с помощью файла учетных данных GSERVICEACCOUNTFILE
        spreadsheet = gc.open_by_key(SHEETID_PARAM)                 # Открываем таблицу Google Sheets по ключу SHEETID_PARAM
        worksheet = spreadsheet.worksheet(sheet_name)               # Получаем лист таблицы по имени sheet_name
        worksheet.append_row(history)                                  # Добавляем строку данных в конец листа
    except Exception as e:
        logger.error(f'save_data_to_google_sheets: {e}')


def set_report_into_gsh():
    # print(f'set_report_into_gsh: Полный путь к файлу:{os.path.abspath(GSERVICEACCOUNTFILE)}')
    # if os.path.isfile(GSERVICEACCOUNTFILE):
    #     print(f"File exists {os.path.abspath(GSERVICEACCOUNTFILE)}")
    # else:
    #     print(f"File does not exist {os.path.abspath(GSERVICEACCOUNTFILE)}")

    try:
        gc = pygsheets.authorize(service_file=GSERVICEACCOUNTFILE)  # Авторизуемся в Google Sheets с помощью файла учетных данных GSERVICEACCOUNTFILE
        sh = gc.open_by_key(SHEETID_PARAM)                          # Открываем таблицу Google Sheets по ключу SHEETID_PARAM
        # Заголовки столбцов
        columns_users = [                                           # Создаем список columns_users с заголовками столбцов таблицы "Пользователи"
            'id', 'tg_id', 'e_mail', 'first_name', 'last_name', 'username', 'last_interaction', 'num_queries',
            'last_dialog', 'last_question', 'last_answer'
        ]

        columns_history = [                                         # Создаем список columns_history с заголовками столбцов таблицы "Оценки"
            'user_id', 'score_name', 'score_text', 'score_chunks', 'score', 'num_token', 'score_time',
            'time_duration'
        ]

        sheet_name = ['Пользователи', 'Оценки']                     # Создаем список sheet_name с названиями листов таблицы

        df_users = get_df_users()[columns_users]                    # Получаем DataFrame df_users и выбираем только нужные столбцы columns_users
        df_users['num_queries'] = df_users['num_queries'].astype(int)   # Преобразовываем столбец num_queries в целочисленный тип данных
        # df_users = df_users.sort_values(by='last_interaction', ascending=True)
        df_score = get_df_history()[columns_history]                # Получаем DataFrame df_score и выбираем только нужные столбцы columns_history
        df_score['time_duration'] = df_score['time_duration'].astype(int)
        wks_write = sh.worksheet_by_title(sheet_name[0])            # Получаем лист таблицы по имени sheet_name[0]
        wks_write.set_dataframe(df_users, (1, 1), encoding='utf-8', fit=True)   # Записываем DataFrame df_users в лист таблицы
        wks_write = sh.worksheet_by_title(sheet_name[1])
        wks_write.set_dataframe(df_score, (1, 1), encoding='utf-8', fit=True)
    except Exception as e:
        logger.error(f'set_report_into_gsh: {e}')


def set_users_into_gsh():
    """
    Добавление нового пользователя в таблицу Google
    :return:
    """
    #print(f'1. set_users_into_gsh: Полный путь к файлу "{GSERVICEACCOUNTFILE}": {os.path.abspath(GSERVICEACCOUNTFILE)}\n {SHEETID_PARAM =}')
    # if os.path.isfile(GSERVICEACCOUNTFILE):
    #     print(f"File exists {os.path.abspath(GSERVICEACCOUNTFILE)}")
    # else:
    #     print(f"File does not exist {os.path.abspath(GSERVICEACCOUNTFILE)}")

    try:
        gc = pygsheets.authorize(service_file=GSERVICEACCOUNTFILE)
        sh = gc.open_by_key(SHEETID_PARAM)
        # Заголовки столбцов
        columns_users = [
            'id', 'tg_id', 'e_mail', 'first_name', 'last_name', 'username', 'last_interaction', 'num_queries'
        ]
        df_users = get_df_users()[columns_users]
        df_users['num_queries'] = df_users['num_queries'].astype(int)
        wks_write = sh.worksheet_by_title('Пользователи')
        wks_write.set_dataframe(df_users, (1, 1), encoding='utf-8', fit=True)
    except Exception as e:
        logger.error(f'set_users_into_gsh: {e}')


def get_report():
    """
    Формируем отчет в электронную таблицу XLS
    :return:
    """

    # Заголовки столбцов
    columns_users = [
        'id', 'tg_id', 'e_mail', 'first_name', 'last_name', 'username', 'last_interaction', 'num_queries'
    ]

    columns_history = [
        'user_id', 'score_name', 'score_text', 'score', 'score_chunck', 'num_token', 'cost_request', 'score_time', 'time_duration'
    ]

    sheet_name = ['Пользователи', 'Оценки']
    sheet_col_width = [
        {'A:A': 12, 'B:B': 20, 'C:C': 20, 'D:D': 20, 'E:E': 30, 'F:F': 20, 'G:G': 20, 'I:I': 20},
        {'A:A': 20, 'B:B': 14, 'C:C': 80, 'D:D': 12, 'E:E': 20, 'F:F': 20, 'G:G': 20, 'I:I': 20}
        ]

    df_users = get_df_users()[columns_users]
    df_score = get_df_history()[columns_history]
    name_report = f'report_{LOGGING_SERVICE}_{datetime.utcnow().strftime("%d.%m.%Y_%H.%M.%S")}.xlsx'
    with pd.ExcelWriter(name_report) as writer:
        workbook = writer.book
        df_users.to_excel(writer, sheet_name=sheet_name[0], index=False)
        df_score.to_excel(writer, sheet_name=sheet_name[1], index=False)
        cell_format = workbook.add_format({'align': 'left', 'text_wrap': 'true'})
        for i, sh_status in enumerate(sheet_name):
            sheet = writer.sheets[sh_status]
            for key in sheet_col_width[i].keys():
                sheet.set_column(str(key), int(sheet_col_width[i][key]), cell_format)
    return name_report


if __name__ == '__main__':

    # url_sh = creat_new_google_sheets('agent500')
    # print(f'{url_sh =}')
    # TODO: выдача прав работает некорректно: после того, как создали новую таблицу ее SHEETID_PARAM сначала надо внести в CONFIF, а уже затем вторым запуском расшаривать
    share_google_sheets('') # выдача права на редактривание каждому у кого есть ссылка

    #get_report()
    #set_users_into_gsh()
    #pass