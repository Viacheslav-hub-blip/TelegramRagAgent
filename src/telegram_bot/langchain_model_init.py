from src.telegram_bot.config import GIGACHAT_API_PERS
import os
from langchain_gigachat.chat_models import GigaChat

os.environ[
    "GIGACHAT_API_PERS"] = GIGACHAT_API_PERS

model_for_answer = GigaChat(verify_ssl_certs=False,
                            credentials=GIGACHAT_API_PERS,
                            temperature=0.8,
                            model="GigaChat-2")

model_for_brief_content = GigaChat(verify_ssl_certs=False,
                                   credentials=GIGACHAT_API_PERS,
                                   temperature=0.8,
                                   model="GigaChat-2")
