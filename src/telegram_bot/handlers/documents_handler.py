import asyncio
import os
import random
from pathlib import Path
from typing import List
from src.telegram_bot.config import some_questions_for_examples, TAVILY_API_KEY
# AIOGRAM
from aiogram import Router, F, Bot
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
# LANGCHAIN
from langchain_community.tools import TavilySearchResults
# TELEGRAMBOT
from src.telegram_bot.create_bot import bot
# SERVICES
from src.telegram_bot.services.llm_model_service import LLMModelService
from src.telegram_bot.services.vectore_store_service import VecStoreService
from src.telegram_bot.services.retriever_service import RetrieverSrvice
from src.telegram_bot.services.documents_saver_service import DocumentsSaver
from src.telegram_bot.services.pdf_reader_service import PDFReader
# AGENT
from src.telegram_bot.langchain_model_init import model_for_brief_content

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["NUMBA_NUM_THREADS"] = "1"
DOWNLOAD_PATH = "/src/telegram_bot/temp_downloads"

router = Router()
tool = TavilySearchResults(k=3)
llm_model_service = LLMModelService(model_for_brief_content)


class LoadFile(StatesGroup):
    file_path = State()
    language = State()
    file_name = State()
    process_file = State()


def _get_content(input_format, language, file_path) -> str:
    """Извлекает содержимое из документа"""
    file_reader = PDFReader(file_path)
    content = file_reader.get_cleaned_content()
    return content


def _save_doc_content(input_format: str, file_path: str, language: List[str], user_id: str,
                      file_name: str) -> str:
    """Сохраняет извлеченную информацию"""
    content = _get_content(input_format, language, file_path)
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    vecstore_store_service = VecStoreService(llm_model_service, retriever, content, file_name)
    summarize_content = vecstore_store_service.save_docs_and_add_in_retriever()
    return summarize_content


async def _save_file(file_id) -> str:
    """Сохраняет загруженный файл"""
    destination = rf"/home/alex/PycharmProjects/pythonProject/src/temp_downloads/{file_id}.pdf"
    await Bot.download(bot, file_id, destination)
    return destination


@router.message(lambda message: message.document is not None)
async def handle_file(message: Message, state: FSMContext):
    if message.document.mime_type == "application/pdf":
        file_id = message.document.file_id
        res_destination = await _save_file(file_id)
        await state.update_data(file_path=res_destination)
        await state.update_data(file_name=message.document.file_name)
        await state.set_state(LoadFile.language)
        await message.answer(f"🌍Выберите язык фаила. На данный момент я поддерживаю:\neng\nrus")
    else:
        await message.reply("Извините, пока я работаю только с PDF файлами📝")


@router.message(F.text, LoadFile.language)
async def choose_file_language(message: Message, state: FSMContext):
    if message.text.lower() in ["eng", "rus"]:
        await state.update_data(language=message.text.lower())
        await message.answer(
            "Теперь мне осталось прочитать файл, чтобы ответить на ваши вопросы💡. Пожалуйста, подождите, это может занять несколько минут\n"
            "Когда я закончу обработку, я напишу. Пока я читаю файл, вы можете продолжать задавать мне вопросы.\n\n"
            f"Например: {some_questions_for_examples[random.randint(0, len(some_questions_for_examples) - 1)]}"
        )
        await state.set_state(LoadFile.process_file)

        loop = asyncio.get_event_loop()
        data = await state.get_data()
        result = await loop.run_in_executor(None,
                                            _save_doc_content,
                                            'pdf',
                                            data.get("file_path"),
                                            [data.get("language")],
                                            str(message.from_user.id),
                                            data.get("file_name"))
        await message.answer(result)
        await state.clear()
    else:
        await message.answer("Я пока не поддерживаю этот язык")


@router.message(Command("clear_documents"))
async def clear_documents(msg: Message):
    VecStoreService.clear_vector_stores(str(msg.from_user.id))
    DocumentsSaver.delete_file_with_files_ids_names(str(msg.from_user.id))
    DocumentsSaver.clear_user_directory(str(msg.from_user.id))
    await msg.answer("Все загруженные документы удалены🗑")
