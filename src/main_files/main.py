import os
import time
from pathlib import Path
from pprint import pprint
from typing import List

from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from src.file_reader import FileReader, SummarizeContentAndDocs
from src.graphs.simpleRag.agent import RagAgent
from langchain_core.language_models.chat_models import BaseChatModel
from src.custon_multivec_retriever import CustomMultiVecRetriever
from transformers import AutoTokenizer
from src.langchain_model_init import model
from src.LangChainVec import get_retriever
from langchain_community.tools.tavily_search import TavilySearchResults


class VecStoreStoreCreator:
    def __init__(self, file_reader: FileReader,
                 text_splitter: TextSplitter,
                 llm_model: BaseChatModel,
                 retriever: CustomMultiVecRetriever,
                 additional_docs: List[Document]) -> None:
        self.file_reader = file_reader
        self.text_splitter = text_splitter
        self.llm_model = llm_model
        self.retriever = retriever
        self.additional_docs = additional_docs

    def _get_markdown_doc_content(self) -> str:
        result_read = self.file_reader.get_content()
        result_markdown = result_read.document.export_to_markdown()
        # print("MAIN----------RESULT MARKDOWN----------")
        # pprint(result_markdown)
        return result_markdown

    def _get_split_documents(self) -> List[str]:
        split_docs = self.text_splitter.split_text(self._get_markdown_doc_content())
        # print("MAIN----------SPLIT DOCS----------")
        # print('count docs', len(split_docs))
        return split_docs

    def _get_summary_doc_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        summary_docs = self.file_reader.get_summarize_docs_content(split_docs, self.llm_model)
        # print(summary_docs)
        # print("MAIN----------RESULT SUMMARY----------")
        # print('count sum docs', len(summary_docs.summary))
        # pprint(summary_docs.summary)
        return summary_docs

    def add_docs_in_retriever(self) -> List[str]:
        """Добавлет документы в векторную базу  возвращает
        краткое содержание
        """
        split_docs = self._get_split_documents()
        result_summary = self._get_summary_doc_content(split_docs)

        doc_ids = [str(i) for i in range(len(split_docs))]
        summarize_docs = [
            Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) for i, summary in
            enumerate(result_summary.summary)
        ]

        self.retriever.vectorstore.add_documents(summarize_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, split_docs)))

        doc_add_is = [str(i) for i in range(len(split_docs), len(self.additional_docs) + 1)]
        self.retriever.vectorstore.add_documents(self.additional_docs)
        self.retriever.docstore.mset(list(zip(doc_add_is, self.additional_docs)))
        return result_summary.summary


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

    retriever = get_retriever()

    vecstore_store_creator = VecStoreStoreCreator(file_reader, text_splitter, model, retriever, additional_docs)
    summary = vecstore_store_creator.add_docs_in_retriever()

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
