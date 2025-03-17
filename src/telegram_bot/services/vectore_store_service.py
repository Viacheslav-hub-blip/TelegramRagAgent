import os
import re
from typing import List, NamedTuple
from uuid import uuid4
from langchain_text_splitters import TextSplitter
from langchain.schema.document import Document
from src.file_reader import FileReader
from src.telegram_bot.services.custon_multivec_retriever import CustomMultiVecRetriever
from src.telegram_bot.services.llm_model_service import LLMModelService, SummarizeContentAndDocs
from src.telegram_bot.services.documents_saver_service import DocumentsSaver

documentSaver = DocumentsSaver()


class SummDocsWithIdsAndSource(NamedTuple):
    summarize_docs_with_ids: List[Document]
    doc_ids: List[str]
    source_docs: List[str]


class VecStoreService:
    def __init__(self, file_reader: FileReader,
                 text_splitter: TextSplitter,
                 model_service: LLMModelService,
                 retriever: CustomMultiVecRetriever,
                 ) -> None:
        self.file_reader = file_reader
        self.text_splitter = text_splitter
        self.model_service = model_service
        self.retriever = retriever

    def _get_markdown_doc_content(self) -> str:
        result_read = self.file_reader.get_content()
        result_markdown = result_read.document.export_to_markdown()
        return result_markdown

    def _get_split_documents(self) -> List[str]:
        content = self._get_markdown_doc_content()
        print("RAW content", content)
        cleaned_content = re.sub(r'\s+|<!-- image -->', ' ', content)
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content)
        print("CLEAN content", cleaned_content)
        cleaned_split_docs = self.text_splitter.split_text(cleaned_content)
        return cleaned_split_docs

    def _get_summary_doc_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        summary_docs = self.model_service.get_summarize_docs_content(split_docs)
        return summary_docs

    def _get_summary_doc_with_ids(self) -> SummDocsWithIdsAndSource:
        cleaned_split_docs = self._get_split_documents()
        result_summary = self._get_summary_doc_content(cleaned_split_docs)
        doc_ids = [str(uuid4()) for _ in range(len(result_summary.summary_with_questions))]
        summarize_docs = [
            Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) for i, summary in
            enumerate(result_summary.summary_with_questions)
        ]
        return SummDocsWithIdsAndSource(summarize_docs, doc_ids, cleaned_split_docs)

    def add_docs_from_reader_in_retriever(self) -> List[str]:
        """Добавлет документы в векторную базу  возвращает
        краткое содержание
        """
        summarize_docs_with_ids, doc_ids, source_docs = self._get_summary_doc_with_ids()
        print(summarize_docs_with_ids, doc_ids, source_docs)
        self.retriever.vectorstore.add_documents(summarize_docs_with_ids)

        user_id = self.retriever.vectorstore._collection_name[5:]
        documentSaver.save_source_docs_in_files(user_id, doc_ids, source_docs)

        result_summary_without_questions = [re.sub(r'Вопросы:.*?(?=\n\n|\Z)', '', summ.page_content, flags=re.DOTALL) for summ in
                                            summarize_docs_with_ids]
        return result_summary_without_questions
