from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder


def ease_link_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text="Мой Telegram", url="https://t.me/Viacheslav_Talks")],
        [InlineKeyboardButton(text="Мой Habr", url="https://habr.com/ru/users/Viacheslav-hub/")],

    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)


def choose_file_or_context_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text="Выбрать файл🗃", callback_data="choose_file_for_search")],
        [InlineKeyboardButton(text="Продолжить работу с ответом📗", callback_data="use_context")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)


def stop_working_with_context_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text="Прекратить работать с ответом🗑", callback_data="stop_working_with_context")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)


def faq_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text='О нас 👀', callback_data='faq')]
    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)
