import re
from typing import List, NamedTuple
from uuid import uuid4
from langchain.schema.document import Document
from src.file_reader import FileReader
import chromadb
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
    def __init__(self,
                 model_service: LLMModelService,
                 retriever: CustomMultiVecRetriever,
                 content: str
                 ) -> None:
        self.model_service = model_service
        self.retriever = retriever
        self.content = content

    def _get_summary_doc_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        """Создает сжатые документы из полных фрагментов
        Если документ всег один(его длина была слшком маленькой для разделения,он остается без изменений)
        Иначе получаем SummarizeContentAndDocs с сжатыми документами и исходными
        """
        if len(split_docs) == 1:
            return SummarizeContentAndDocs([self.model_service.get_summarize_docs(split_docs)], split_docs)
        return self.model_service.get_summarize_docs_with_questions(split_docs)

    def _add_metadata_in_docs(self, summarized_docs: list[str]) -> (list[Document], list[str]):
        """Добавлет metadata в документы: уникальный id документа, принадлежность к группе и позицию документа
        в группе. Сделано для дальнейшей возможности извлечения соседних документов"""
        doc_ids, docs_section = [str(uuid4()) for _ in range(len(summarized_docs))], str(uuid4())
        summarize_docs_with_metadata = [
            Document(
                page_content=summary,
                metadata={"doc_id": doc_ids[i], "belongs_to": docs_section, "doc_number": i}) for i, summary in
            enumerate(summarized_docs)
        ]
        return summarize_docs_with_metadata, doc_ids

    def _get_summary_doc_with_metadata(self) -> SummDocsWithIdsAndSource:
        """Возвращает сжатые документы с дополнительными данными, id документов
        и исходные документы
        """
        source_split_documents: list[str] = TextSplitterService.get_split_documents(self.content)
        summarized_docs: SummarizeContentAndDocs = self._get_summary_doc_content(source_split_documents)
        docs_with_metadata, doc_ids = self._add_metadata_in_docs(summarized_docs.summary_texts)
        return SummDocsWithIdsAndSource(docs_with_metadata, doc_ids, source_split_documents)

    def get_documents_without_add_questions(self, documents: list[Document]) -> list[str]:
        """Удаляет из сжатых текстов дополнительные вопросы, которые были добавлены перед векторизацией"""
        documents_without_questions = [re.sub(r'Вопросы:.*?(?=\n\n|\Z)', '', summ.page_content, flags=re.DOTALL)
                                       for summ in
                                       documents]
        return documents_without_questions

    def add_docs_from_reader_in_retriever(self) -> List[str]:
        """Добавлет документы в векторную базу и возвращает
        краткое содержание без дополнитльно созданных вопросов
        """
        summarize_docs_with_ids, doc_ids, source_docs = self._get_summary_doc_with_metadata()
        user_id = self.retriever.vectorstore._collection_name[5:]
        self.retriever.vectorstore.add_documents(summarize_docs_with_ids)
        documentSaver.save_source_docs_in_files(user_id, doc_ids, source_docs)
        return self.get_documents_without_add_questions(summarize_docs_with_ids)

    @staticmethod
    def clear_vector_stores(user_id: str):
        """Удаляет векторное хранилище пользователя""""
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            shutil.rmtree(f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
            client.clear_system_cache()
