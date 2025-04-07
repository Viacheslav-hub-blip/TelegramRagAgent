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
from src.telegram_bot.services.RetrieverService import RetrieverSrvice
from src.telegram_bot.services.documents_saver_service import DocumentsSaver
from src.file_reader import FileReader
# AGENT
from src.telegram_bot.langchain_model_init import model

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["NUMBA_NUM_THREADS"] = "1"

router = Router()
DOWNLOAD_PATH = "/src/telegram_bot/temp_downloads"

tool = TavilySearchResults(k=3)
llm_model_service = LLMModelService(model)


class LoadFile(StatesGroup):
    file_path = State()
    language = State()
    file_name = State()
    process_file = State()


def _get_content(input_format, language, file_path):
    file_reader = FileReader(
        input_format=input_format,
        tessdata_path="/usr/share/tesseract-ocr/5/tessdata/",
        file_path=Path(file_path),
        language=language,
        generate_picture_images=False
    )
    markdown_content = file_reader.get_markdown()
    content = file_reader.get_cleaned_content(markdown_content)
    return content


def _save_summarize_doc_content(input_format: str, file_path: str, language: List[str], user_id: str,
                                file_name: str) -> str:
    content = _get_content(input_format, language, file_path)
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    vecstore_store_service = VecStoreService(llm_model_service, retriever, content, file_name)
    summarize_content = vecstore_store_service.save_docs_and_add_in_retriever()
    return summarize_content


async def _save_file(file_id) -> str:
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
        await message.answer(f"Выберите язык фаила. На данный момент я поддерживаю:\neng\nrus")
    else:
        await message.reply("Извините, пока я работаю только с PDF файлами")


@router.message(F.text, LoadFile.language)
async def choose_file_language(message: Message, state: FSMContext):
    if message.text.lower() in ["eng", "rus"]:
        await state.update_data(language=message.text.lower())
        await message.answer(
            "Супер! Теперь мне осталось прочитать файл, чтобы ответить на ваши вопросы. Пожалуйста, подождите\n"
            "Когда я закончу обработку, я напишу. Пока я читаю файл, вы можете продолжать задавать мне вопросы.\n\n"
            f"Например: {some_questions_for_examples[random.randint(0, len(some_questions_for_examples) - 1)]}"
        )
        await state.set_state(LoadFile.process_file)

        loop = asyncio.get_event_loop()
        data = await state.get_data()
        result = await loop.run_in_executor(None,
                                            _save_summarize_doc_content,
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
    try:
        print(str(msg.from_user.id))
        VecStoreService.clear_vector_stores(str(msg.from_user.id))
        DocumentsSaver.clear_user_directory(str(msg.from_user.id))
        DocumentsSaver.delete_file_with_files_ids_names(str(msg.from_user.id))
        await msg.answer("Все загруженные документы удалены")
    except:
        await msg.answer("По некоторым причинам мне сейчас не удалось удалить все документы")
