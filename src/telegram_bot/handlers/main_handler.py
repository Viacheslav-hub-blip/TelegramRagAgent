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
from src.telegram_bot.services.retriever_service import RetrieverSrvice
from src.telegram_bot.services.documents_saver_service import DocumentsSaver
from src.telegram_bot.services.documents_getter_service import DocumentsGetterService
# AGENT
from src.telegram_bot.langchain_model_init import model_for_answer
from src.agents.RagAgents.pro_version import RagAgent

router = Router()
DOWNLOAD_PATH = "/src/telegram_bot/temp_downloads"


class AgentAnswer(NamedTuple):
    question: str
    generation: str
    source_docs_names: list
    answer_without_retrieve: bool
    file_metadata_id: str


class ChooseFileForSearch(StatesGroup):
    possible_files_names = State()
    select_file_names = State()
    file_was_selected = State()
    selected_name = State()


def _format_answer(answer: AgentAnswer) -> str:
    names = "\n".join(answer.source_docs_names)
    if answer.answer_without_retrieve:
        res_ans = (
            f"{answer.generation}\n\n"
            f"Мне не удалось найти ответ на этот вопрос в документах☹"
        )
        return res_ans
    else:
        if answer.file_metadata_id:
            res_ans = (
                f"{answer.generation}\n\n"
                f"Поиск был в установленном документе:\n"
                f"{names}"
            )
            return res_ans

        res_ans = (
            f"{answer.generation}\n\n"
            f"Для поиска я использовал документы:\n"
            f"{names}"
        )
        return res_ans


def _invoke_agent(user_id: str, question: str, file_metadata_id: str = None) -> AgentAnswer:
    retriever = RetrieverSrvice.get_or_create_retriever(user_id)
    rag_agent = RagAgent(model=model_for_answer, retriever=retriever)
    result = rag_agent().invoke({"question": question, "user_id": user_id, "file_metadata_id": file_metadata_id})
    question, generation, = result["question"], result["answer"]

    answer_without_retrieve = result["answer_without_retrieve"]
    if answer_without_retrieve:
        used_docs_names = []
    else:
        used_docs_names = result["used_docs"]
    print('file metadata id', file_metadata_id)
    return AgentAnswer(question, generation, used_docs_names, answer_without_retrieve, file_metadata_id)


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
        f"Привет!👋\nЯ ассистент, который поможет тебе работать с документами📝"
        "\nДля начала работы просто отправьте файл"
        "\nЕсли у вас нет файла, то просто зайде мне любой вопрос и я на него отвечу\n\n"
        f"Например: {some_questions_for_examples[random.randint(0, len(some_questions_for_examples) - 1)]}",
        reply_markup=faq_kb()
    )


@router.callback_query(F.data == "faq")
async def faq_handler(callback: CallbackQuery):
    await callback.message.answer("Вот ифнормация о нас", reply_markup=ease_link_kb())


@router.callback_query(F.data == "choose_file_for_search")
async def choose_file_for_search(callback: CallbackQuery, state: FSMContext):
    all_files_ids_names = DocumentsGetterService.get_files_ids_names(str(callback.from_user.id))
    names = '\n'.join([name for id, name in all_files_ids_names.items()])
    await callback.message.answer(f"💡Введите название одного файла из загруженных:\n"
                                  f"{names}\n\n"
                                  f"✨или напишите 'искать во всех'")
    await state.update_data(possible_files_names=names)
    await state.set_state(ChooseFileForSearch.select_file_names)


@router.message(F.text, ChooseFileForSearch.select_file_names)
async def select_file_name_for_search(msg: Message, state: FSMContext):
    possible_names = await state.get_data()
    possible_names = possible_names.get("possible_files_names")
    msg_text = msg.text.lower().strip()
    if msg_text in possible_names:
        await state.set_state(ChooseFileForSearch.file_was_selected)
        await state.update_data(selected_name=msg.text)
        await msg.answer(f"Выбран файл для поиска: {msg.text}")
    elif msg_text == 'искать во всех':
        await state.clear()
        await msg.answer(f"Поиск по всем файлам включен📖")
    else:
        await msg.answer("Не могу найти такой файл, попробуйте еще раз😔")


async def _get_file_for_search_id(msg: Message, state: FSMContext) -> str | None:
    file_for_search = await state.get_data()
    file_for_search_name = file_for_search.get("selected_name")
    file_for_search_id = None
    if file_for_search_name:
        file_for_search_id = [k for k, v in DocumentsGetterService.get_files_ids_names(str(msg.from_user.id)).items() if
                              v == file_for_search_name][0]
    return file_for_search_id


async def _exist_loaded_docs(msg: Message) -> Message:
    exist_loaded_docs = DocumentsSaver.check_exist_user_directory(str(msg.from_user.id))
    if exist_loaded_docs:
        send_message = await msg.answer(f"Пожалуйста, подождите, я печатаю.\n"
                                        f"Кстати, у вас есть загруженные документы\n"
                                        f"🗑Если вы хотите их удалить, напишите '/clear_documents'")
    else:
        send_message = await msg.answer(f"Пожалуйста, подождите, я печатаю✍")
    return send_message


@router.message()
async def any_message_handler(msg: Message, state: FSMContext):
    send_message: Message = await _exist_loaded_docs(msg)
    if await state.get_state() == 'ChooseFileForSearch:file_was_selected':
        file_for_search_id = await _get_file_for_search_id(msg, state)
        answer = _invoke_agent(str(msg.from_user.id), msg.text, file_for_search_id)
    else:
        answer = _invoke_agent(str(msg.from_user.id), msg.text)

    format_ans = _format_answer(answer)
    await bot.delete_message(chat_id=msg.chat.id, message_id=send_message.message_id)
    await msg.answer(format_ans, reply_markup=choose_file_for_search_kb())
