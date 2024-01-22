from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from typing import List
from config import TEMPERATURE
from core.gpt_models import GptModel
from core.utils import Singleton
from create_bot import OPENAI_API_KEY
from logger.logger import logger

class ChatGPTClient(metaclass=Singleton):
    client: AsyncOpenAI

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY
        )

    async def send_request(self, model: GptModel, messages: List[ChatCompletionMessageParam]) -> ChatCompletion:
        logger.info(f'Запрос к ChatGPT, модель: {model.name}')
        completion = await self.client.chat.completions.create(
            model=model.name,
            messages=messages,
            temperature=TEMPERATURE
        )
        logger.info('Конец запроса к ChatGPT')

        return completion
