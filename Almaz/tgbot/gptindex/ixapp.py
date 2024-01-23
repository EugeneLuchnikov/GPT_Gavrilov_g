import requests
import zipfile
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain_openai import OpenAIEmbeddings
from ixconfig import ixconfig
from config import config

def load_file(url: str):
    """ Функция загрузки документа по url как текст."""
    try:
        response = requests.get(url)
        # Проверка ответа и если была ошибка - формирование исключения.
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(e)

def load_file_content(url: str):
    response = requests.get(url) # Получение документа по url.
    response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
    return response.content

def download_file_from_url(url, save_path):
    content= load_file_content(url)
    with open(save_path, 'wb') as f:
        f.write(content)
        print(f"downloaded and saved as {save_path}")

def load_zip(url: str, path_dest:str, file_dest:str):
    """ Функция загружает векторную Базу знаний."""
    dest = os.path.join(path_dest, file_dest)+".zip"
    if not os.path.exists(dest):
        response = requests.get(url)
        # Проверка ответа и если была ошибка - формирование исключения.
        response.raise_for_status()
        # Сохранение архива.
        with open(dest, 'wb') as file:
            file.write(response.content)
    # Разархивирование Базы знаний.
    extract_dest = os.path.join(path_dest, file_dest)
    if not os.path.exists(extract_dest):
        with zipfile.ZipFile(dest, 'r') as zip:
            zip.extractall(extract_dest)
    # Загрузка векторной Базы знаний.

class ixapp_class:
    config = ixconfig
    prefixname = "federallab_bd_"
    db = None
    question = None
    promt = None

    def init(self):
        self.db = self.load_db(self.config.url_bd, "index")
        self.question = self.load_db(self.config.url_question, "question")
        self.promt = self.load_file(self.config.url_promt, "prompt.txt")

    def load_file(self, url:str, file_name:str):
        path_name = os.path.join(config.index_path,file_name)
        if os.path.exists(path_name):
            with open(path_name,"r") as f:
                text = f.read()
        else:
            text = load_file(url)
            with open(path_name,"w") as f:
                f.write(text)
        return text
    def load_db(self, url:str, name:str):
        db_name = f"{self.prefixname}{name}"
        load_zip(url, config.index_path, db_name)
        extract_dest = os.path.join(config.index_path, db_name)
        if os.path.exists(os.path.join(extract_dest, db_name)):
            extract_dest = os.path.join(extract_dest, db_name)
        # print(extract_dest)
        db = FAISS.load_local(extract_dest, OpenAIEmbeddings())
        return db
    def bd_retriever(self, query):
        docs = self.db.similarity_search_with_score(query, k=3)
        message_content = ''
        for i, doc in enumerate(docs):
            message_content = message_content + \
                f'Отрывок документа №{i+1}:{doc[0].page_content}\n'
        return message_content
    def find_query(self, query):
        db = self.question
        doc = db.similarity_search_with_score(query, k=1)
        if doc[0][1] < 0.08:
            return True, doc[0][0].metadata['answer_gpt']
        else:
            return False, ''


ixapp = ixapp_class()

