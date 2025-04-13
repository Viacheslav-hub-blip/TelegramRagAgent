import asyncio
import logging

from aiogram import Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from create_bot import bot
from handlers.main_handler import router as main_router
from handlers.documents_handler import router as docs_router


async def main():
    print(
        len(" но бурал вечерка но такие времена что уже молодежь не работает увольняются поэтому вот работают уже зайду даже вместо зарплаты"))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_routers(docs_router, main_router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
