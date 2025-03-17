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
from src.telegram_bot.services.text_splitter_service import TextSplitterService

documentSaver = DocumentsSaver()


class SummDocsWithIdsAndSource(NamedTuple):
    summarize_docs_with_ids: List[Document]
    doc_ids: List[str]
    source_docs: List[str]


class VecStoreService:
    def __init__(self, file_reader: FileReader,
                 model_service: LLMModelService,
                 retriever: CustomMultiVecRetriever,
                 ) -> None:
        self.file_reader = file_reader
        self.model_service = model_service
        self.retriever = retriever

    def _get_markdown_doc_content(self) -> str:
        result_read = self.file_reader.get_content()
        result_markdown = result_read.document.export_to_markdown()
        return result_markdown

    def _get_cleaned_markdown_doc_content(self) -> str:
        content = self._get_markdown_doc_content()
        cleaned_content = re.sub(r'\s+|<!-- image -->', ' ', content)
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content)
        return cleaned_content

    def _get_split_documents(self) -> List[str]:
        cleaned_content = self._get_cleaned_markdown_doc_content()
        print("len cleaned_content", len(cleaned_content))
        if len(cleaned_content) <= 1500:
            return [cleaned_content]
        elif 1500 < len(cleaned_content) <= 6000:
            text_splitter = TextSplitterService(chunk_size=500, chunk_overlap=100).create_text_splitter()
            cleaned_split_docs = text_splitter.split_text(cleaned_content)
            return cleaned_split_docs
        else:
            text_splitter = TextSplitterService(chunk_size=700, chunk_overlap=150).create_text_splitter()
            cleaned_split_docs = text_splitter.split_text(cleaned_content)
            return cleaned_split_docs

    def _get_summary_doc_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        if len(split_docs) == 1:
            return SummarizeContentAndDocs([self.model_service.get_summarize_docs(split_docs)], split_docs)
        return self.model_service.get_summarize_docs_with_questions(split_docs)

    def _get_summary_doc_with_ids(self) -> SummDocsWithIdsAndSource:
        cleaned_split_docs = self._get_split_documents()
        result_summary = self._get_summary_doc_content(cleaned_split_docs)
        print("count docs", len(cleaned_split_docs), "SPLIT DOCS", cleaned_split_docs)
        doc_ids = [str(uuid4()) for _ in range(len(result_summary.summary_texts))]
        docs_section = str(uuid4())
        summarize_docs = [
            Document(
                page_content=summary,
                metadata={"doc_id": doc_ids[i], "belongs_to": docs_section, "doc_number": i}) for i, summary in
            enumerate(result_summary.summary_texts)
        ]
        return SummDocsWithIdsAndSource(summarize_docs, doc_ids, cleaned_split_docs)

    def add_docs_from_reader_in_retriever(self) -> List[str]:
        """Добавлет документы в векторную базу  возвращает
        краткое содержание
        """
        summarize_docs_with_ids, doc_ids, source_docs = self._get_summary_doc_with_ids()
        self.retriever.vectorstore.add_documents(summarize_docs_with_ids)

        user_id = self.retriever.vectorstore._collection_name[5:]
        documentSaver.save_source_docs_in_files(user_id, doc_ids, source_docs)

        result_summaries_without_questions = [re.sub(r'Вопросы:.*?(?=\n\n|\Z)', '', summ.page_content, flags=re.DOTALL)
                                              for summ in
                                              summarize_docs_with_ids]
        return result_summaries_without_questions
