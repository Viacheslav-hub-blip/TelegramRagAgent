import os
from typing import List
from uuid import uuid4
from langchain_text_splitters import TextSplitter
from langchain.schema.document import Document
from src.file_reader import FileReader
from src.telegram_bot.services.custon_multivec_retriever import CustomMultiVecRetriever
from src.telegram_bot.services.llm_model_service import LLMModelService, SummarizeContentAndDocs
from src.telegram_bot.services.documents_saver_service import DocumentsSaver

documentSaver = DocumentsSaver()


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
        split_docs = self.text_splitter.split_text(self._get_markdown_doc_content())
        return split_docs

    def _get_summary_doc_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        summary_docs = self.model_service.get_summarize_docs_content(split_docs)
        return summary_docs

    def add_docs_in_retriever(self) -> List[str]:
        """Добавлет документы в векторную базу  возвращает
        краткое содержание
        """
        split_docs = self._get_split_documents()
        result_summary = self._get_summary_doc_content(split_docs)
        doc_ids = [str(uuid4()) for _ in range(len(split_docs))]
        summarize_docs = [
            Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) for i, summary in
            enumerate(result_summary.summary)
        ]
        self.retriever.vectorstore.add_documents(summarize_docs)
        user_id = self.retriever.vectorstore._collection_name
        documentSaver.save_source_docs_in_files(user_id, doc_ids, split_docs)
        return result_summary.summary