import re
import os
import sys
import asyncio
import tiktoken
from typing import Any, Dict, List
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings

from config import (
    ROOT_DIR, 
    SETTINGS_PATH, 
    FAISS_DB_DIR, 
    SYSTEM_PROMPT_FILE, 
    USER_PROMPT_FILE,
    CHUNKS_PROMPT_FILE,
    MODEL, TEMPERATURE, 
    TELEGRAM_MAX_MESSAGE_LENGTH,
    DIALOG_ALGORITHM,
    DialogAlgorithm
)
from create_bot import OPENAI_API_KEY
from dbase.models import History
from logger.logger import logger

if OPENAI_API_KEY is not None:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

faiss_db_dir = os.path.join(ROOT_DIR, FAISS_DB_DIR)
os.chdir(faiss_db_dir)


class WorkerOpenAI:
    def __init__(self, faiss_db_dir=faiss_db_dir, list_indexes=None, mod=MODEL):
        # старт инициализации промптов
        system_prompt_file = os.path.join(ROOT_DIR, SETTINGS_PATH, SYSTEM_PROMPT_FILE)
        system_prompt = ""
        try:
            with open(system_prompt_file, 'r', encoding='utf-8') as file:
                system_prompt = file.read()
                logger.info(f'(Прочитали system_prompt)')
        except Exception as e:
            print(f'Ошибка чтения system_prompt: {e}')

        user_prompt_file = os.path.join(ROOT_DIR, SETTINGS_PATH, USER_PROMPT_FILE)
        user_prompt = ""
        try:
            with open(user_prompt_file, 'r', encoding='utf-8') as file:
                user_prompt = file.read()
                logger.info(f'(Прочитали user_prompt)')
        except Exception as e:
            print(f'Ошибка чтения user_prompt: {e}')

        chunks_prompt_file = os.path.join(ROOT_DIR, SETTINGS_PATH, CHUNKS_PROMPT_FILE)
        chunks_prompt = ""
        try:
            with open(chunks_prompt_file, 'r', encoding='utf-8') as file:
                chunks_prompt = file.read()
                logger.info(f'(Прочитали chunks_prompt)')
        except Exception as e:
            print(f'Ошибка чтения chunks_prompt: {e}')
        # конец инициализации промптов

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
        # системные настройки
        self.chat_manager_system = system_prompt
        self.chat_manager_user = user_prompt
        self.chat_manager_chunks = chunks_prompt

        def create_search_index(indexes) -> FAISS:
            """
                Чтение индексов из всех индексных файлов
                :param path: локальный путь в проекте до папки с индексами
                :return: база индексов
                """
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
            return db

        # Если База данных embedding уже создана ранее
        # print(f'Проверим путь до базы знаний: {faiss_db_dir}')
        #if faiss_db_dir:
            # print(f'{os.getcwd() = }')
            # print("Ищем готовую базу данных. Путь: ", faiss_db_dir)
            # print("Курсы: ", self.list_indexes)
        
        self.search_index = create_search_index(self.list_indexes)

        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY
        )

    # пример подсчета токенов
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(self, messages: List[ChatCompletionMessageParam]) -> int:
        """Return the number of tokens used by a list of messages."""
        model = self.model.name
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
            return self.num_tokens_from_messages(messages)
        elif "gpt-4" in model:
            logger.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages)
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

    async def get_chatgpt_answer(self, topic: str, history_items: List[History]):
        # Выборка документов по схожести с вопросом
        docs = await self.search_index.asimilarity_search(topic, k=8)
        #print(f'get_chatgpt_answer: {docs}')
        doc_chunks = re.sub(r'\n{2}', ' ', '\n '.join(
            [f'\n==  ' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
        
        # system_prompt = self.chat_manager_system.format(
        #     max_tokens = self.model.max_tokens_for_answer,
        #     max_characters = TELEGRAM_MAX_MESSAGE_LENGTH,
        #     doc_chunks=doc_chunks
        # )

        # user_prompt = self.chat_manager_user.format(topic=topic, doc_chunks=doc_chunks)

        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt}
        # ]

        # messages = self.add_previous_messages(messages, history_items)

        if DIALOG_ALGORITHM == DialogAlgorithm.SEPARATE_MESSAGES:
            messages = self.generate_dialog_in_separate_messages(topic, history_items, doc_chunks)
        elif DIALOG_ALGORITHM == DialogAlgorithm.SINGLE_MESSAGE:
            messages = self.generate_dialog_in_single_message(topic, history_items, doc_chunks)
        else:
            raise Exception(f'Неизвестный алгоритм диалога: {DIALOG_ALGORITHM}')

        # TODO: добавить вторую более дешевую модель. Выбирать модель в зависимости от объема передаваемого user_prompt
        logger.info(f'Иcпользуем модель: {self.model.name}')
        completion = await self.client.chat.completions.create(
            model=self.model.name,
            messages=messages,
            temperature=TEMPERATURE
        )

        #print(f'{completion.usage.completion_tokens =}')
        #print('ЦЕНА запроса с ответом :', 0.004*(completion.usage.total_tokens/1000), ' $')
        #print('===========================================: \n')
        #print('Ответ ChatGPT: ')
        #print(completion.choices[0].message.content)
        #cost_request = 0.02020202
        if completion.usage is not None:
            cost_request = self.model.input_price*(completion.usage.prompt_tokens/1000) + self.model.output_price*(completion.usage.completion_tokens/1000)
        else:
            cost_request = 0

        logger.info(f'ЦЕНА запроса с ответом :  {cost_request}$')
        return completion, messages, docs, cost_request
    
    # def add_previous_messages(self, messages: List[Dict[str, str]], history_items: List[History]) -> List[Dict[str, str]]:
    #     new_messages = []

    #     new_messages.append(messages[0])

    #     for item in history_items:
    #         new_messages.append({"role": "user", "content": item.question})
    #         new_messages.append({"role": "assistant", "content": item.answer})

    #     new_messages.append(messages[1])

    #     num_tokens = self.num_tokens_from_messages(new_messages)
    #     while num_tokens > MAX_TOKENS_FOR_REQUEST and len(new_messages) > 2:
    #         del new_messages[1:3]
    #         num_tokens = self.num_tokens_from_messages(new_messages)

    #     return new_messages

    def generate_dialog_in_separate_messages(self, topic: str, history_items: List[History], doc_chunks: str) -> List[ChatCompletionMessageParam]:
        new_messages = []

        system_prompt = self.chat_manager_system.format(
            max_tokens = self.model.max_tokens_for_answer,
            max_characters = TELEGRAM_MAX_MESSAGE_LENGTH,
        ) + "\n" + self.chat_manager_chunks.format(doc_chunks=doc_chunks)

        user_prompt = topic

        new_messages.append({"role": "system", "content": system_prompt})

        for item in history_items:
            new_messages.append({"role": "user", "content": item.question})
            new_messages.append({"role": "assistant", "content": item.answer})

        new_messages.append({"role": "user", "content": user_prompt})

        num_tokens = self.num_tokens_from_messages(new_messages)
        while num_tokens > self.model.max_tokens_for_request and len(new_messages) > 2:
            del new_messages[1:3]
            num_tokens = self.num_tokens_from_messages(new_messages)

        return new_messages
    
    def generate_dialog_in_single_message(self, topic: str, history_items: List[History],
            doc_chunks: str) -> List[ChatCompletionMessageParam]:
        
        history_item_count = len(history_items)
        
        new_messages = self.generate_dialog_in_single_message_for_history_part(topic, history_items, doc_chunks, history_item_count)
        num_tokens = self.num_tokens_from_messages(new_messages)

        while num_tokens > self.model.max_tokens_for_request and history_item_count > 0:
            history_item_count -= 1
            new_messages = self.generate_dialog_in_single_message_for_history_part(topic, history_items, doc_chunks, history_item_count)
            num_tokens = self.num_tokens_from_messages(new_messages)

        return new_messages
    
    def generate_dialog_in_single_message_for_history_part(
            self, topic: str, history_items: List[History],
            doc_chunks: str, history_item_count: int) -> List[ChatCompletionMessageParam]:
        
        new_messages = []

        items = history_items[-history_item_count:]

        dialog = "\n".join([f'Клиент: {item.question}\nКонсультант: {item.answer}' for item in items])

        system_prompt = self.chat_manager_system.format(
            max_tokens = self.model.max_tokens_for_answer,
            max_characters = TELEGRAM_MAX_MESSAGE_LENGTH,
        )

        user_prompt = self.chat_manager_user.format(
            topic = topic,
            dialog = dialog,
            doc_chunks = doc_chunks
        )

        new_messages.append({"role": "system", "content": system_prompt})
        new_messages.append({"role": "user", "content": user_prompt})

        return new_messages
    

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
