from aiogram import Router, types, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command, CommandStart
from src.telegram_bot.keyboards.all_kb import main_kb, create_rat_kb
from src.telegram_bot.keyboards.inline_kbs import ease_link_kb, faq_kb

router = Router()


@router.message(CommandStart())
async def start_handler(msg: Message):
    await msg.answer(
        "Привет! Я чат бот, который поможет тебе работать  с документами с помощью GigaChat.\n Для начала работы просто отправьте фаил",
        reply_markup=faq_kb()
    )


@router.callback_query(F.data == "faq")
async def start_handler(callback: CallbackQuery):
    await callback.message.answer("Вот ифнормация о нас", reply_markup=ease_link_kb())


@router.message()
async def ant_message_handler(msg: Message):
    await msg.answer(f"Твой ID: {msg.from_user.id}",
                     reply_markup=create_rat_kb()
                     )
