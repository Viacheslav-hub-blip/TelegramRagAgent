import asyncio
import os
from pathlib import Path
from pprint import pprint
from typing import List, NamedTuple
# AIOGRAM
from aiogram import Router, F, Bot
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
# LANGCHAIN
from langchain_community.tools import TavilySearchResults
# TELEGRAMBOT
from src.telegram_bot.keyboards.inline_kbs import ease_link_kb, faq_kb
from src.telegram_bot.create_bot import bot
# SERVICES
from src.telegram_bot.services.llm_model_service import LLMModelService
from src.telegram_bot.services.text_splitter_service import TextSplitterService
from src.telegram_bot.services.vectore_store_service import VecStoreService
from src.telegram_bot.services.RetrieverService import RetrieverSrvice
from src.telegram_bot.services.documents_saver_service import DocumentsSaver
from src.file_reader import FileReader
# AGENT
from src.langchain_model_init import model
from src.graphs.simpleRag.agent import RagAgent

os.environ["TAVILY_API_KEY"] = "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"
os.environ["NUMBA_NUM_THREADS"] = "1"
router = Router()
DOWNLOAD_PATH = "/src/telegram_bot/temp_downloads"

tool = TavilySearchResults(k=3)
text_splitter = TextSplitterService().create_text_splitter()
llm_model_service = LLMModelService(model)


class AgentAnswer(NamedTuple):
    question: str
    generation: str
    web_search: str
    source_docs: list


class LoadFile(StatesGroup):
    file_path = State()
    language = State()
    process_file = State()


def _collect_source_links(source_docs: list, max_links: int = 2) -> list:
    source_links = []
    for doc in source_docs:
        for link in doc.metadata["source"]:
            source_links.append(link)
    return source_links[:max_links]


def _format_answer(answer: AgentAnswer) -> str:
    if answer.web_search not in ['No', 'Нет', 'Нет', 'NO']:
        links = "\n".join(_collect_source_links(answer.source_docs))
        res_ans = (
            f"Мой ответ:\n {answer.generation}\n"
            f"Для ответа я использовал интернет ресурсы:\n"
            f"{links}")
        return res_ans
    return f"{answer.generation}"


def _invoke_agent(user_id: str, question: str) -> AgentAnswer:
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    rag_agent = RagAgent(model=model, retriever=retriever, web_search_tool=tool)
    result = rag_agent().invoke({"question": question})
    question, generation, web_search = result["question"], result["generation"], result["web_search"]
    try:
        documents = result["documents"]
        pprint(result["documents"])
    except:
        documents = []
    return AgentAnswer(question, generation, web_search, documents)


def _save_summarize_doc_content(input_format: str, file_path: str, language: List[str], user_id: str) -> List[str]:
    file_reader = FileReader(
        input_format=input_format,
        tessdata_path="/usr/share/tesseract-ocr/5/tessdata/",
        file_path=Path(file_path),
        language=language,
        generate_picture_images=False
    )
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    vecstore_store_service = VecStoreService(file_reader, text_splitter, llm_model_service, retriever)
    summarize_content = vecstore_store_service.add_docs_in_retriever()
    return summarize_content


def get_old_users_ids() -> List[str]:
    with open("/home/alex/PycharmProjects/pythonProject/src/users_ids") as f:
        ids = f.readlines()
    return ids


def write_new_ids(ids: List[str]) -> None:
    with open("/home/alex/PycharmProjects/pythonProject/src/users_ids", "w") as f:
        f.writelines(ids)


async def _save_file(file_id) -> str:
    destination = rf"/home/alex/PycharmProjects/pythonProject/src/temp_downloads/{file_id}.pdf"
    await Bot.download(bot, file_id, destination)
    return destination


@router.message(CommandStart())
async def _start_handler(msg: Message):
    old_users_ids = get_old_users_ids()
    if str(msg.from_user.id) not in old_users_ids:
        old_users_ids.append(str(msg.from_user.id))
        write_new_ids(old_users_ids)
    await msg.answer(
        "Привет!\nЯ чат бот, который поможет тебе работать с документами с помощью GigaChat!"
        "\nДля начала работы просто отправьте файл"
        "\nЕсли у вас нет файла, то просто зайде мне любой вопрос и я на него отвечу",
        reply_markup=faq_kb()
    )


@router.message(Command("clear_documents"))
async def clear_documents(msg: Message):
    try:
        print(str(msg.from_user.id))
        RetrieverSrvice.clear_retriever(str(msg.from_user.id))
        DocumentsSaver().clear_user_directory(str(msg.from_user.id))
        await msg.answer("Все загруженные документы удалены")
    except:
        await msg.answer("По некоторым причинам мне сейчас не удалось удалить все документы")


@router.message(lambda message: message.document is not None)
async def handle_file(message: Message, state: FSMContext):
    if message.document.mime_type == "application/pdf":
        file_id = message.document.file_id
        res_destination = await _save_file(file_id)
        await state.update_data(file_path=res_destination)
        await state.set_state(LoadFile.language)
        await message.answer(f"Выберите язык фаила. На данный момент я поддерживаю:\neng\nrus")
    else:
        await message.reply("Извините, пока я работаю только с PDF файлами")


@router.message(F.text, LoadFile.language)
async def choose_file_language(message: Message, state: FSMContext):
    if message.text in ["eng", "rus"]:
        await state.update_data(language=message.text)
        await message.answer(
            "Супер! Теперь мне осталось прочитать файл, чтобы ответить на ваши вопросы. Пожалуйста, подождите\n"
            "Пока я читаю файл, вы можете продолжать задавать мне вопросы. Когда я закончу обработку, я напишу")
        await state.set_state(LoadFile.process_file)

        loop = asyncio.get_event_loop()
        data = await state.get_data()
        result = await loop.run_in_executor(None,
                                            _save_summarize_doc_content,
                                            'pdf',
                                            data.get("file_path"),
                                            [data.get("language")],
                                            str(message.from_user.id))
        await message.answer("".join(result)[:4000])
        await state.clear()
    else:
        await message.answer("Я пока не поддерживаю этот язык")


@router.callback_query(F.data == "faq")
async def start_handler(callback: CallbackQuery):
    await callback.message.answer("Вот ифнормация о нас", reply_markup=ease_link_kb())


@router.message()
async def any_message_handler(msg: Message):
    exist_loaded_docs = DocumentsSaver.check_exist_user_directory(str(msg.from_user.id))
    if exist_loaded_docs:
        send_message = await msg.answer(f"Пожалуйста, подождите, я печатаю.\n"
                                        f"Кстати, у вас есть загруженные документы\n"
                                        f"Если вы хотите их удалить, напишите '/clear_documents'")
    else:
        send_message = await msg.answer(f"Пожалуйста, подождите, я печатаю")
    answer = _invoke_agent(str(msg.from_user.id), msg.text)
    format_ans = _format_answer(answer)
    await bot.delete_message(chat_id=msg.chat.id, message_id=send_message.message_id)
    await msg.answer(format_ans)
