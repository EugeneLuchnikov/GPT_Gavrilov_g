from dotenv import  load_dotenv
from config import config
import os
from pprint import pprint
def init_config():
    config.TG_TOKEN = os.getenv("TG_TOKEN")
    config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    config.database_path = os.getenv("database_path")
    config.config_file_path = ""
    config.data_path = os.getenv("data_path")
    config.index_path = os.getenv("index_path")

def load_config():
    """Начальная точка входа, загрузка делается в самом модуле один раз при импорте!!!"""
    pass

load_dotenv(".env")
init_config()
def ppo(obj):
    # Получаем словарь атрибутов объекта
    attributes = vars(obj)  # или obj.__dict__

    # Проходим по атрибутам в цикле
    for key, value in attributes.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    ppo(config)



