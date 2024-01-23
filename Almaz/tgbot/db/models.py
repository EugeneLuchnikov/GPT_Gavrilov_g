from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
import pytz

from sqlalchemy import Column, BigInteger, Integer, String, ForeignKey, Date, Time, Boolean, Float, DATETIME
from datetime import datetime as ddatetime, timezone, timedelta

def now():
    return ddatetime.now(timezone(timedelta(hours=3))).replace(tzinfo=None)

class Base(AsyncAttrs, DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)  # ID от Telegram
    username = Column(String(80))
    tg_id = Column(BigInteger, nullable=False, unique=True)
    e_mail = Column(String(80))
    first_name = Column(String(64))
    last_name = Column(String(64))
    dt_create = Column(DATETIME, default=now)
    dt_startquota =Column(DATETIME)
    quota = Column(Integer)
    dialog = Column(Integer)



class UserMessage(Base):
    __tablename__ = 'user_messages'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer)  # ID чата в Telegram
    user_id = Column(Integer, ForeignKey('users.id'))
    u_message = Column(String())
    u_message_id = Column(Integer)
    a_message = Column(String())
    a_message_id = Column(Integer)
    u_tokens = Column(Integer)
    a_tokens = Column(Integer)
    dt_create = Column(DATETIME, default=now)
    dt_answer = Column(DATETIME)
    dt_rating = Column(DATETIME)
    time_duration = Column(Float)
    rating = Column(Integer)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)

class ErrorLog(Base):  # фиксация ошибок
    __tablename__ = 'error_logs'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer)  # ID чата в Telegram
    user_id = Column(Integer, ForeignKey('users.id'))
    user_message_id = Column(Integer, ForeignKey('user_messages.id'))

    dt_create = Column(DATETIME, default=now)
    function_name = Column(String())  # имя функции, которая вызвала ошибку
    arguments = Column(String())  # аргументы, переданные в функцию
    error_message = Column(String())  # сообщение об ошибке

class Rating(Base):    # оценка ответов
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.id'))  # ID пользователя в телеграмм
    message_id = Column(Integer)  # ID сообщения в телеграмм
    user_message_id = Column(Integer, ForeignKey('user_messages.id'))
    rating = Column(Integer)
    dt_create = Column(DATETIME, default=now)

class TextMessage(Base):            # служебные сообщения
    __tablename__ = 'text_messages'
    id = Column(Integer, primary_key=True)
    messages_name = Column(String(100))
    text = Column(String)

class ModelPrompt(Base):           # промты
    __tablename__ = 'model_prompts'
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    system = Column(String())
    assistant = Column(String())
    prompt_length = Column(Integer)
