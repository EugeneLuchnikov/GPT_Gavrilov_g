from abc import ABC, abstractmethod 
from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from dbase.models import History
from core.chatgpt_client import ChatGPTClient
from core.gpt_models import GptModel
from core.utils import load_setting_file, num_tokens_from_messages
from logger.logger import logger
from config import (
    DIALOG1_SYSTEM_FILE,
    DIALOG1_USER_FILE,
    DIALOG2_SYSTEM_FILE,
    DIALOG2_USER_FILE,
    DIALOG3_SYSTEM_FILE,
    DIALOG3_USER_FILE,
    DIALOG3_SUM_SYSTEM_FILE,
    DIALOG3_SUM_USER_FILE,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    DIALOG_ALGORITHM,
    DialogAlgorithm
)


class DialogBuilder(ABC):
    @abstractmethod
    def load_prompts(self) -> None:
        pass

    @abstractmethod
    async def create_messages(self, model: GptModel, topic: str, history_items: List[History], doc_chunks: str) -> List[ChatCompletionMessageParam]:
        pass

    @classmethod
    def get_builder(cls) -> 'DialogBuilder':
        builder = None
        if DIALOG_ALGORITHM == DialogAlgorithm.SEPARATE_MESSAGES:
            builder = DialogBuilderForSeparateMessages()
        elif DIALOG_ALGORITHM == DialogAlgorithm.SINGLE_MESSAGE:
            builder = DialogBuilderForSingleMessage()
        elif DIALOG_ALGORITHM == DialogAlgorithm.SUMMARIZATION:
            builder = DialogBuilderForSummarization()
        else:
            raise Exception(f'Неизвестный алгоритм диалога: {DIALOG_ALGORITHM}')
        return builder


class DialogBuilderForSeparateMessages(DialogBuilder):
    system_prompt: str
    user_prompt: str

    def load_prompts(self) -> None:
        self.system_prompt = load_setting_file(DIALOG1_SYSTEM_FILE)
        self.user_prompt = load_setting_file(DIALOG1_USER_FILE)

    async def create_messages(self, model: GptModel, topic: str, history_items: List[History], doc_chunks: str) -> List[ChatCompletionMessageParam]:
        new_messages = []

        system_prompt = self.system_prompt.format(
            max_tokens = model.max_tokens_for_answer,
            max_characters = TELEGRAM_MAX_MESSAGE_LENGTH,
            doc_chunks=doc_chunks
        )

        user_prompt = self.user_prompt.format(topic = topic)

        new_messages.append({"role": "system", "content": system_prompt})

        for item in history_items:
            new_messages.append({"role": "user", "content": item.question})
            new_messages.append({"role": "assistant", "content": item.answer})

        new_messages.append({"role": "user", "content": user_prompt})

        num_tokens = num_tokens_from_messages(new_messages, model.name)
        while num_tokens > model.max_tokens_for_request and len(new_messages) > 2:
            del new_messages[1:3]
            num_tokens = num_tokens_from_messages(new_messages, model.name)

        return new_messages


class DialogBuilderForSingleMessage(DialogBuilder):
    system_prompt: str
    user_prompt: str

    def load_prompts(self):
        self.system_prompt = load_setting_file(DIALOG2_SYSTEM_FILE)
        self.user_prompt = load_setting_file(DIALOG2_USER_FILE)

    async def create_messages(self, model: GptModel, topic: str, history_items: List[History], doc_chunks: str) -> List[ChatCompletionMessageParam]:
        history_item_count = len(history_items)
        
        new_messages = self.create_messages_for_history_part(model, topic, history_items, doc_chunks, history_item_count)
        num_tokens = num_tokens_from_messages(new_messages, model.name)

        while num_tokens > model.max_tokens_for_request and history_item_count > 0:
            history_item_count -= 1
            new_messages = self.create_messages_for_history_part(model, topic, history_items, doc_chunks, history_item_count)
            num_tokens = num_tokens_from_messages(new_messages, model.name)

        return new_messages
    
    def create_messages_for_history_part(self, model: GptModel, topic: str, history_items: List[History],
        doc_chunks: str, history_item_count: int) -> List[ChatCompletionMessageParam]:
        
        new_messages = []

        items = history_items[-history_item_count:]

        dialog = "\n".join([f'Клиент: {item.question}\nКонсультант: {item.answer}' for item in items])

        system_prompt = self.system_prompt.format(
            max_tokens = model.max_tokens_for_answer,
            max_characters = TELEGRAM_MAX_MESSAGE_LENGTH,
        )

        user_prompt = self.user_prompt.format(
            topic = topic,
            dialog = dialog,
            doc_chunks = doc_chunks
        )

        new_messages.append({"role": "system", "content": system_prompt})
        new_messages.append({"role": "user", "content": user_prompt})

        return new_messages


class DialogBuilderForSummarization(DialogBuilder):
    system_prompt: str
    user_prompt: str
    sum_system_prompt: str
    sum_user_prompt: str
    client: ChatGPTClient

    def __init__(self) -> None:
        self.client = ChatGPTClient()

    def load_prompts(self) -> None:
        self.system_prompt = load_setting_file(DIALOG3_SYSTEM_FILE)
        self.user_prompt = load_setting_file(DIALOG3_USER_FILE)
        self.sum_system_prompt = load_setting_file(DIALOG3_SUM_SYSTEM_FILE)
        self.sum_user_prompt = load_setting_file(DIALOG3_SUM_USER_FILE)

    async def create_messages(self, model: GptModel, topic: str, history_items: List[History], doc_chunks: str) -> List[ChatCompletionMessageParam]:
        new_messages = []

        summarized_dialog = await self.summarize_dialog(model, history_items)
        dialog = "\n".join([f'Клиент: {item.question}\nКонсультант: {item.answer}' for item in history_items])
        logger.info(f'Исходный диалог:\n{dialog}')
        logger.info(f'Саммаризированный диалог:\n{summarized_dialog}')

        system_prompt = self.system_prompt.format(
            max_tokens = model.max_tokens_for_answer,
            max_characters = TELEGRAM_MAX_MESSAGE_LENGTH,
        )

        user_prompt = self.user_prompt.format(
            topic = topic,
            summarized_dialog = summarized_dialog,
            doc_chunks = doc_chunks
        )

        new_messages.append({"role": "system", "content": system_prompt})
        new_messages.append({"role": "user", "content": user_prompt})

        return new_messages

    async def summarize_dialog(self, model: GptModel, history_items: List[History]) -> str:
        history_item_count = len(history_items)

        if history_item_count == 0:
            return ''
        
        new_messages = self.create_messages_for_summarization(history_items, history_item_count)
        num_tokens = num_tokens_from_messages(new_messages, model.name)

        while num_tokens > model.max_tokens_for_request and history_item_count > 0:
            history_item_count -= 1
            new_messages = self.create_messages_for_summarization(history_items, history_item_count)
            num_tokens = num_tokens_from_messages(new_messages, model.name)

        completion = await self.client.send_request(model, new_messages)

        summarized_dialog = completion.choices[0].message.content

        return summarized_dialog if summarized_dialog else ''
    
    def create_messages_for_summarization(self, history_items: List[History], history_item_count: int) -> List[ChatCompletionMessageParam]:
        new_messages = []

        items = history_items[-history_item_count:]

        dialog = "\n".join([f'Клиент: {item.question}\nКонсультант: {item.answer}' for item in items])

        system_prompt = self.sum_system_prompt
        user_prompt = self.sum_user_prompt.format(dialog = dialog)

        new_messages.append({"role": "system", "content": system_prompt})
        new_messages.append({"role": "user", "content": user_prompt})

        return new_messages