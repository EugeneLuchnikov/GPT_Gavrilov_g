from aiogram import types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State
from gptindex.ixapp import ixapp
from  openai import AsyncOpenAI
from datetime import datetime
from bot.keyboard import create_rating_keyboard
from db.botdb import BotDB, BotDBData
from utils.logger import logger
from gptindex.gptutils import num_tokens_from_string
from config import config

class Gpt_helper:
    prompt_tokens=0
    completion_tokens=0
    def __init__(self, message: types.Message, context:FSMContext):
        self.message = message
        self.query = message.text
        self.context = context
        self.context_data = None
        self.api_key = None
        self.clientOpenAI =None
        self.quota = 0
        self.api_key = None
        self.q_tokens = 0
        self.a_tokens = 0
        self.data:BotDBData = None
        self.db:BotDB = None #BotDB(self.message)
        self.d1 = datetime.now()
    async def close(self):
        await self.db.close()
    async def get_data(self):
        if self.context:
            self.context_data = await self.context.get_data()
        else:
            self.context_data = {}
        d = self.context_data
        self.data = BotDBData()
        self.db= BotDB(self.message, self.data)
        await self.db.update_user()
        if "messages" in d:
            self.data.messages = d["messages"]
        if "api_key" in d:
            self.data.api_key = d["api_key"]
    async def update_data(self):
        await self.db.update_data()
        await self.context.update_data(messages=self.data.messages, api_key = self.data.api_key)

    async def save_api_key(self):
        self.set_api_key(self.query)
        await self.update_data()


    def set_api_key(self, api_key:str):
        api_key = api_key.strip()
        if api_key=="0":
            api_key=None
        self.data.api_key = api_key
        self.quota = 0

    async def answer(self, text):
        await self.message.answer( text=text, reply_markup=types.ReplyKeyboardRemove())

    async def pause(self):
        self.delta_d1 = datetime.now()

        self.pause_message = await self.message.answer("...▌")
        self.topic_message = self.pause_message

    def ClientAI(self):
        api_key = self.api_key
        if not api_key:
            api_key = config.OPENAI_API_KEY
        if not self.clientOpenAI:
            self.clientOpenAI = AsyncOpenAI(api_key=api_key)
        return self.clientOpenAI

    async def query_refiner(self,  query):
        config = ixapp.config
        model = config.openai_model
        client:AsyncOpenAI = self.ClientAI()
        conversation = self.conversation_string( self.data.messages, config.dialog_depth)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": config.refiner_prompt.replace('{conversation}', conversation).replace('{query}',query)},
                {"role": "user", "content": f"{query}"}
            ],
            temperature=0,
            max_tokens=1256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        u = response.usage
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        return str(response.choices[0].message.content)

    def conversation_string(self, messages, dialog_depth):
        conversation_string = ''
        k = 0 if len(messages) <= dialog_depth * \
                 2 else -1 * dialog_depth * 2
        for message in messages[k:]:
            conversation_string += f"{message['role']}: {message['content']}\n"
        return conversation_string

    async def edit_text_rating(self, text):
        await self.pause_message.edit_text(text, reply_markup= create_rating_keyboard(self.db.usermessage_id))
    async def edit_text(self, text, printsec=True, wait=False):
        d2=datetime.now()
        sec = (d2-self.d1).total_seconds()
        t = text
        if printsec:
            t+=f"...({sec}с)"
        if wait:
            if (d2 - self.delta_d1).total_seconds() > 1.7:
                wait = False

        if not wait:
            self.delta_d1 = d2
            logger.debug(t)
            await self.pause_message.edit_text(t)
            logger.debug("send")

    async def print_usage_tokens(self):
        d2 = datetime.now()
        sec = (d2 - self.d1).total_seconds()
        if self.api_key:
            rest = ""
        else:
            rest = f" осталось {self.db.data.quota}"
        text = f"использовано. вопрос: {self.prompt_tokens}, ответ:{self.completion_tokens} токенов ({sec}с) {rest}"
        # await db.update_message(message, text)
        await self.message.answer(text=text)


    async def completion(self, usercontent, refined_query):
        client: AsyncOpenAI = self.ClientAI()
        config = ixapp.config
        stream = config.stream
        full_response = ""
        if config.testing:
            full_response = "Ответ системы"
        else:
            usercontent = f"Документ с информацией для ответа пользователю:\n {usercontent}.\nВопрос клиента: {refined_query}"
            responses = await client.chat.completions.create(
                    model=config.openai_model,
                    messages=[
                        {"role": "system", "content": ixapp.promt},
                        {"role": "user",
                         "content": usercontent}
                    ],
                    temperature=0,
                    stream=stream,
            )
            if stream:
                async for response in responses:
                    delta = (response.choices[0].delta.content or "")
                    if len(delta)>0:
                        full_response +=delta
                        await self.edit_text(full_response + "▌", wait=True)
                self.prompt_tokens += num_tokens_from_string(ixapp.promt + usercontent + 6)
                self.completion_tokens +=  num_tokens_from_string( full_response)
            else:
                full_response = responses.choices[0].message.content
                usage = responses.usage
                self.prompt_tokens += usage.prompt_tokens
                self.completion_tokens +=  usage.completion_tokens

        await self.edit_text_rating(text=full_response)
        await self.db.answer_usermessage(self.topic_message.message_id, full_response, self.prompt_tokens, self.completion_tokens)
        await self.print_usage_tokens()
        return full_response

    async def check_quota(self):
        from datetime import timedelta
        if self.api_key:
            return True
        if self.db.data.quota<=0:
            text = f"Вы превысили количество токенов {self.db.data.quota}, счетчик обновится {self.db.data.dt_startquota + timedelta(hours=10)}"
            m = await self.message.answer(text)
            await self.db.answer_usermessage(m.message_id, text)
            return False
        return True

    async def answer_gpt(self):
        await self.db.new_usermessage(self.query)
        h = self
        if not await self.check_quota():
            return
        await h.pause()
        try:
            is_query, db_response = ixapp.find_query(h.query)
            logger.debug(f"Вопрос: {h.query}  Dialog {h.data.dialog}")
            if h.data.dialog == 0:
                h.data.messages = []



            if not is_query:
                # Формирование уточняющего вопроса по диалогу.
                if len(h.data.messages) > 1:
                    await h.edit_text("Уточняю вопрос...")
                    refined_query = await h.query_refiner(h.query)
                    logger.debug(f"refined_query: {refined_query}")
                    await h.edit_text(f"Отвечаю на вопрос: {refined_query}")
                else:
                    refined_query = h.query
                h.data.messages.append({"role": "user", "content": f"{h.query} ({refined_query})"})
                # Возвращение релевантных документов по запросу.
                content = ixapp.bd_retriever(refined_query)
                full_response = await h.completion(content, refined_query)
                logger.debug("Ответ: "+full_response)

                h.data.messages.append({"role": "assistant", "content": f"{h.query} ({full_response})"})
            else:
                text = f"(ответ из базы вопросов)\n{db_response}"
                logger.debug(text)
                await self.db.answer_usermessage( h.topic_message.message_id, text, self.prompt_tokens, self.completion_tokens)
                if ixapp.config.askrating:
                    await h.edit_text_rating(text)
                h.data.messages.append(
                    {"role": "user", "content": f"{h.query}"})
                h.data.messages.append(
                    {"role": "assistant", "content": db_response})
            await h.update_data()
        except Exception as e:
            text = "Ошибка: "+str(e)
            logger.exception("answer_gpt")
            await h.answer(text)
            await self.log("answer_gpt", "", "Ошибка: "+text)

    async def select_choice(self, state: State, text):
        logger.debug(text)
        await self.context.set_state(state)
        await self.message.answer( text=text, reply_markup=types.ReplyKeyboardRemove())

    async def log(self, func, param, text):
        await self.db.add_log(func, param, text)

async def get_gpt_helper(message, state):
    h = Gpt_helper(message, state)
    await h.get_data()
    return h


async def answer_gpt(message: types.Message, context:FSMContext):
    h = Gpt_helper(message, context)
    await h.answer_gpt()

