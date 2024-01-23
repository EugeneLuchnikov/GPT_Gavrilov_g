from db.base import async_session, init_database
from db.models import  User, now
from  utils.logger import logger
from sqlalchemy import select, engine
from db.botdb import BotDB, types


class Test:
    user_tg_id = 409521+4
    username = "Test"
    user_id = 0

    logger.catch
    async def test1(self):
        print("dfasdf")
        session = async_session()
        u=User
        q: engine.Result = await session.execute(
            select(u.id, u.quota, u.dt_startquota ).where(u.tg_id == self.user_tg_id)
        )
        u = self
        r = q.first()
        if r:
            u.user_id, u.quota, u.dt_startquota = r
        dt = now()

        if not r:
            user = User(
                tg_id=u.user_tg_id,
                username=u.username,
                quota = 10**4,
                dt_startquota =dt
            )
            session.add(user)
            await session.commit()
            self.user_id = user.id
        print(u.dt_startquota)
        print(dt)
        if u.dt_startquota == None or (dt - u.dt_startquota).seconds > 10*3600:
            print("asdfasdf")
        await session.close()


    @logger.catch
    async def test2(self):
        session =  async_session()
        result = await session.execute(
            select(User.id).where(User.tg_id == self.user_tg_id)
        )
        self.user_id = result.scalar_one_or_none()
        u = self
        if not self.user_id:
            user = User(
                tg_id=u.user_tg_id,
                username=u.username,
            )
            session.add(user)
            await session.commit()
            self.user_id = user.id
        await session.close()

    async def test3(self):
        b = BotDB(None)
        b.user_id = self.user_id
        # await b.update_user()
        await b.new_usermessage("Test")
        await b.answer_usermessage(0, "Answer")
        await b.add_rating(2)
        await b.add_log("-","p","Error")
