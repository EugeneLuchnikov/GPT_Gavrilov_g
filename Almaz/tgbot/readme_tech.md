# Установка локально на машину
1. установить виртуальную среду и зависимости
virtualenv env
env/scripts/activate
pip install -r requirements.txt

# Рефакторинг
- создать локальные настройки
- вынести модули по смыслу
- Выносим все зависимости от настроек
- создать начальные значения констант

# Вопросы
Как в Python запустить несколько асинхронных процедур одновременно

import asyncio

async def coroutine1():
    print("Coroutine 1 started")
    await asyncio.sleep(2)
    print("Coroutine 1 finished")

async def coroutine2():
    print("Coroutine 2 started")
    await asyncio.sleep(1)
    print("Coroutine 2 finished")

async def main():
    # Запускаем обе асинхронные процедуры параллельно
    task1 = asyncio.create_task(coroutine1())
    task2 = asyncio.create_task(coroutine2())

    # Ждем завершения обеих задач
    await asyncio.gather(task1, task2)

# Запускаем основную асинхронную программу
asyncio.run(main())