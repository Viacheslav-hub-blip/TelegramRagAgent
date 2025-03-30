import os
import time
from pathlib import Path
from pprint import pprint
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from src.file_reader import FileReader
from src.agents.simpleRag.agent import RagAgent
from transformers import AutoTokenizer
from src.telegram_bot.langchain_model_init import model
from src.telegram_bot.services.RetrieverService import get_or_create_retriever
from langchain_community.tools.tavily_search import TavilySearchResults

from src.telegram_bot.services.llm_model_service import LLMModelService
from src.telegram_bot.services.vectore_store_service import VecStoreService

if __name__ == '__main__':
    t1 = time.time()
    os.environ['HF_TOKEN'] = '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'
    os.environ["TAVILY_API_KEY"] = "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"
    os.environ['GROQ_API_KEY'] = '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'
    os.environ[
        "GIGACHAT_API_PERS"] = "ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmViOTBhNDZmLTAxNzktNDY4Yi04ODljLTc3ZDZhOTA0YmJjZg=="

    additional_docs = [
        Document(
            page_content="Слава Рыльков — молодой разработчик, 20 лет. Живет в Москве, учится в университете. Слава интересуется программированием и языковыми моделями.",
            metadata={"source": "tweet", "doc_id": "2"},
            id=2,
        ),
        Document(
            page_content="Последним проектом SLava стала разработка финансовой платформы и создание умного помощника.",
            metadata={"source": "tweet", "doc_id": "4"},
            id=4,
        )
    ]

    file_reader = FileReader(
        input_format='pdf',
        tessdata_path="/usr/share/tesseract-ocr/5/tessdata/",
        file_path=Path("/home/alex/PycharmProjects/pythonProject/content/gold_fish.pdf"),
        language=["rus"]
    )

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=400, chunk_overlap=100
    )
    user_id = "1234"
    retriever = get_or_create_retriever(user_id)
    llm_model_service = LLMModelService(model)
    vecstore_store_creator = VecStoreService(file_reader, text_splitter, llm_model_service, retriever, additional_docs)
    summary = vecstore_store_creator.save_docs_and_add_in_retriever()

    tool = TavilySearchResults(k=3)
    rag_agent = RagAgent(model=model, retriever=retriever, web_search_tool=tool)
    t2 = time.time()

    # print("---------RETR INVOKE--------")
    # print(retriever.invoke("Слава Рыльков"))
    # print()
    # print("Краткое содержание документа:")
    pprint(summary)
    print()
    print("---------Time--------")
    print("Time before ready agent", t2 - t1)
    while True:
        input_question = input("Введите сообщение: ")

        if input_question != "q":
            inputs = {"question": input_question}
            result = rag_agent().invoke(inputs)

            question, generation, web_search = result["question"], result["generation"], result["web_search"]
            documents: List[Document] = result["documents"]
            print("##QUESTION## ", question)
            print("##ANSWER## ", generation)
            print("##WEB_SERACH## ", web_search)
            pprint(documents)
            print()
        else:
            exit()
