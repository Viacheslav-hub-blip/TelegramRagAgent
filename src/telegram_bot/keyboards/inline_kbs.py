from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder


def ease_link_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text="Мой Telegram", url="https://t.me/Viacheslav_Talks")],
        [InlineKeyboardButton(text="Мой Habr", url="https://habr.com/ru/users/Viacheslav-hub/")],

    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)


def faq_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text='О нас', callback_data='faq')]
    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)
