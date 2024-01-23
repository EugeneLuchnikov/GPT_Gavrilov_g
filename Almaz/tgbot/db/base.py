from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from db.models import Base


#from config import SQLALCHEMY

class db_class:
    def __init__(self, database_path) -> None:
        self.engine = create_async_engine(database_path, echo=True)
        # self.async_session = async_sessionmaker(self.engine)
        self.async_session:sessionmaker = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
    async def async_main(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

# Выносим все зависимости от настроек
database:db_class = None
async_session:sessionmaker = None

def connect_db() -> AsyncSession:
    return async_session()

def init_database(database_path):
    global database
    global async_session
    database = db_class(database_path)
    async_session = database.async_session


