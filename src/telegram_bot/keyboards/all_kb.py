from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def main_kb():
    kb_list = [
        [KeyboardButton(text="О боте"), KeyboardButton(text="Загрузить фаил")]
    ]
    keyboard = ReplyKeyboardMarkup(
        keyboard=kb_list,
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Воспользуйтесь меню:"
    )
    return keyboard


def create_rat_kb():
    """Клавиатура с оценками"""
    builder = ReplyKeyboardBuilder()
    for item in [str(i) for i in range(1, 6)]:
        builder.button(text=item)
    builder.button(text="Назад")
    return builder.as_markup(resize_keyboard=True)
