import os

from langchain_huggingface import HuggingFaceEmbeddings

from src.telegram_bot.config import embeddings_model_name, HF_TOKEN

os.environ['HF_TOKEN'] = HF_TOKEN

embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_name
)
