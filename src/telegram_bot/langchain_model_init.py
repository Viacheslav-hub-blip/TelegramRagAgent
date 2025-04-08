from src.telegram_bot.config import GIGACHAT_API_PERS
import os
from langchain_gigachat.chat_models import GigaChat

os.environ[
    "GIGACHAT_API_PERS"] = GIGACHAT_API_PERS

model = GigaChat(verify_ssl_certs=False,
                 credentials=GIGACHAT_API_PERS,
                 temperature=0.8,
                 model="GigaChat-2")
