import random
from pprint import pprint
from typing import List, NamedTuple
from src.telegram_bot.config import some_questions_for_examples
# AIOGRAM
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.state import StatesGroup, State
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
# TELEGRAMBOT
from src.telegram_bot.keyboards.inline_kbs import ease_link_kb, faq_kb, choose_file_for_search_kb
from src.telegram_bot.create_bot import bot
# SERVICES
from src.telegram_bot.services.RetrieverService import RetrieverSrvice
from src.telegram_bot.services.documents_saver_service import DocumentsSaver
from src.telegram_bot.services.documents_getter_service import DocumentsGetterService
# AGENT
from src.telegram_bot.langchain_model_init import model
from src.agents.RagAgents.pro_version import RagAgent

router = Router()
DOWNLOAD_PATH = "/src/telegram_bot/temp_downloads"


class AgentAnswer(NamedTuple):
    question: str
    generation: str
    web_search: str
    source_docs: list
    source_docs_names: list


class ChooseFileForSearch(StatesGroup):
    possible_files_names = State()
    select_file_names = State()
    selected_name  = State()


def _format_answer(answer: AgentAnswer) -> str:
    if answer not in ['No', 'Нет', 'Нет', 'NO']:
        print(answer.source_docs_names)
        names = "".join(answer.source_docs_names)
        res_ans = (
            f"Мой ответ:\n{answer.generation}\n\n"
            f"Для поиска я использовал документы:\n"
            f"{names}"
        )
        return res_ans
    return f"{answer.generation}"


def _invoke_agent(user_id: str, question: str, file_metadata_name: str) -> AgentAnswer:
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    rag_agent = RagAgent(model=model, retriever=retriever)
    result = rag_agent().invoke({"question": question, "user_id": user_id})
    question, generation, used_docs_names = result["question"], result["answer"], result["used_docs"]
    try:
        documents = result["documents"]
        pprint(result["documents"])
    except:
        documents = []

    return AgentAnswer(question, generation, "", documents, used_docs_names)


def get_old_users_ids() -> List[str]:
    with open("/home/alex/PycharmProjects/pythonProject/src/users_ids") as f:
        ids = f.readlines()
    return ids


def write_new_ids(ids: List[str]) -> None:
    with open("/home/alex/PycharmProjects/pythonProject/src/users_ids", "w") as f:
        f.writelines(ids)


@router.message(CommandStart())
async def _start_handler(msg: Message):
    old_users_ids = get_old_users_ids()
    if str(msg.from_user.id) not in old_users_ids:
        old_users_ids.append(str(msg.from_user.id))
        write_new_ids(old_users_ids)
    await msg.answer(
        f"Привет!\nЯ чат бот, который поможет тебе работать с документами с помощью GigaChat!"
        "\nДля начала работы просто отправьте файл"
        "\nЕсли у вас нет файла, то просто зайде мне любой вопрос и я на него отвечу\n\n"
        f"Например: {some_questions_for_examples[random.randint(0, len(some_questions_for_examples))]}",
        reply_markup=faq_kb()
    )


@router.callback_query(F.data == "faq")
async def faq_handler(callback: CallbackQuery):
    await callback.message.answer("Вот ифнормация о нас", reply_markup=ease_link_kb())


@router.callback_query(F.data == "choose_file_for_search")
async def choose_file_for_search(callback: CallbackQuery, state: FSMContext):
    all_files_ids_names = DocumentsGetterService.get_files_ids_names(str(callback.from_user.id))
    names = '\n'.join([name for id, name in all_files_ids_names.items()])
    await callback.message.answer(f"Введите название одного файла из загруженных:\n"
                                  f"{names}")
    await state.update_data(possible_files_names=names)
    await state.set_state(ChooseFileForSearch.select_file_names)


@router.message(F.text, ChooseFileForSearch.select_file_names)
async def select_file_name_for_search(msg: Message, state: FSMContext):
    possible_names = await state.get_data()
    possible_names = possible_names.get("possible_files_names")

    if msg.text in possible_names:
        await state.update_data(selected_name=msg.text)
        await msg.answer(f"Выбран файл для поиска: {msg.text}")
    else:
        await msg.answer("Не могу найти такой файл, попробуйте еще раз")


@router.message()
async def any_message_handler(msg: Message, state: FSMContext):
    exist_loaded_docs = DocumentsSaver.check_exist_user_directory(str(msg.from_user.id))
    file_for_search = await state.get_data()
    file_for_search = file_for_search.get("selected_name")
    if exist_loaded_docs:
        send_message = await msg.answer(f"Пожалуйста, подождите, я печатаю.\n"
                                        f"Кстати, у вас есть загруженные документы\n"
                                        f"Если вы хотите их удалить, напишите '/clear_documents'")
    else:
        send_message = await msg.answer(f"Пожалуйста, подождите, я печатаю")

    answer = _invoke_agent(str(msg.from_user.id), msg.text, file_for_search)
    format_ans = _format_answer(answer)
    await bot.delete_message(chat_id=msg.chat.id, message_id=send_message.message_id)
    await msg.answer(format_ans, reply_markup=choose_file_for_search_kb())
