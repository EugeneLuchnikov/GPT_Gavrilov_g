import re
import os
import asyncio
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from config import ROOT_DIR, FAISS_DB_DIR, MODEL, NUMBER_OF_CHUNKS
from core.chatgpt_client import ChatGPTClient
from core.dialog_builder import DialogBuilder
from core.utils import Singleton
from create_bot import OPENAI_API_KEY
from dbase.models import History
from logger.logger import logger

if OPENAI_API_KEY is not None:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

faiss_db_dir = os.path.join(ROOT_DIR, FAISS_DB_DIR)
os.chdir(faiss_db_dir)


class WorkerOpenAI(metaclass=Singleton):
    def __init__(self, faiss_db_dir=faiss_db_dir, list_indexes=None, mod=MODEL):
        self.dialog_builder = DialogBuilder.get_builder()
        self.dialog_builder.load_prompts()

        # Составим список всех индексов в папке faiss_db_dir:
        # print(f'Ищем список курсов: {faiss_db_dir}')
        if list_indexes is None:
            list_indexes = []
            for folder in os.listdir(faiss_db_dir):
                if os.path.isdir(os.path.join(faiss_db_dir, folder)):
                    list_indexes.append(os.path.basename(folder))
        #print(f'__init__: Нашли базы: {list_indexes}')

        self.model = mod
        self.list_indexes = list_indexes

        def create_search_index(indexes) -> FAISS:
            """
                Чтение индексов из всех индексных файлов
                :param path: локальный путь в проекте до папки с индексами
                :return: база индексов
                """
            logger.info('Загрузка векторной базы началась')
            db: FAISS = None
            db_path = os.path.join(ROOT_DIR, FAISS_DB_DIR)
            flag = True  # Признак первой базы для чтения. Остальные базы будем добавлять к имеющейся
            # Перебор всех курсов в списке courses:
            # print(f'Старт read_faiss_indexes: {indexes =}')
            count_base = 0  # сосчитаем количество курсов
            for index_file in indexes:
                index_path = os.path.join(db_path, index_file)  # получаем полный путь к курсу
                # print(f'read_faiss_indexes - ищем индекс {count_base}: {index_file =}, {index_path =}')
                count_base += 1
                if flag:
                    # Если flag равен True, то создается база данных FAISS из папки index_path
                    db = FAISS.load_local(index_path, OpenAIEmbeddings())
                    flag = False
                    # print(f'read_faiss_indexes: прочитали новый индекс')
                else:
                    # Иначе происходит объединение баз данных FAISS
                    if db is not None:
                        db.merge_from(FAISS.load_local(index_path, OpenAIEmbeddings()))
                    # print(f'read_faiss_indexes: Добавили в индекс')
            logger.info('Загрузка векторной базы завершена')
            return db

        self.search_index = create_search_index(self.list_indexes)

        self.client = ChatGPTClient()

    async def get_chatgpt_answer(self, topic: str, history_items: List[History]):
        # Выборка документов по схожести с вопросом
        docs = await self.search_index.asimilarity_search(topic, k=NUMBER_OF_CHUNKS)

        doc_chunks = re.sub(r'\n{2}', ' ', '\n '.join(
            [f'\n==  ' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
        
        messages = await self.dialog_builder.create_messages(self.model, topic, history_items, doc_chunks)

        completion = await self.client.send_request(self.model, messages)

        if completion.usage is not None:
            cost_request = self.model.input_price*(completion.usage.prompt_tokens/1000) + self.model.output_price*(completion.usage.completion_tokens/1000)
        else:
            cost_request = 0

        logger.info(f'ЦЕНА запроса с ответом :  {cost_request}$')
        return completion, messages, docs, cost_request
    

if __name__ == '__main__':
    question = """
    Я прохожу урок Треккинга. Расскажи подробнее про Фильтр Калмана.
    """
    # Создаем объект для дообучения chatGPT
    # Если База данных embedding уже создана ранее
    print(f'{os.path.abspath(faiss_db_dir) = } ')
    curator = WorkerOpenAI(faiss_db_dir=faiss_db_dir)
    answer = asyncio.run(curator.get_chatgpt_answer(question, []))
    print(answer)
