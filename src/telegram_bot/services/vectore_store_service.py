import re
import shutil
from typing import List, NamedTuple
from uuid import uuid4
from langchain.schema.document import Document
import chromadb
from src.telegram_bot.services.retriever_service import CustomRetriever
from src.telegram_bot.services.llm_model_service import LLMModelService, SummarizeContentAndDocs
from src.telegram_bot.services.documents_saver_service import DocumentsSaver
from src.telegram_bot.services.text_splitter_service import TextSplitterService


class SummDocsWithSourceAndIds(NamedTuple):
    summarize_docs_with_ids: List[Document]
    doc_ids: List[str]
    source_docs: List[Document]


class VecStoreService:
    def __init__(self,
                 model_service: LLMModelService,
                 retriever: CustomRetriever,
                 content: str,
                 file_name: str
                 ) -> None:
        self.model_service = model_service
        self.retriever = retriever
        self.content = content
        self.file_name = file_name

    def _get_summary_doc_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        """Создает сжатые документы из полных фрагментов
        Если документ всег один(его длина была слшком маленькой для разделения,он остается без изменений)
        Иначе получаем SummarizeContentAndDocs с сжатыми документами и исходными
        """
        if len(split_docs) == 1:
            return SummarizeContentAndDocs([self.model_service.get_summarize_docs(split_docs)], split_docs)
        return self.model_service.get_summarize_docs_with_questions(split_docs)

    def _add_metadata_in_summary_docs(self, doc_ids, docs_section, summarized_docs: list[Document]) -> list[Document]:
        """Добавлет metadata в сжатые документы: уникальный id документа, принадлежность к группе и позицию документа
        в группе. Сделано для дальнейшей возможности извлечения соседних документов"""
        summarize_docs_with_metadata = [
            Document(page_content=doc.page_content,
                     metadata={"doc_id": doc_ids[i], "belongs_to": docs_section, "doc_number": i})
            for i, doc in enumerate(summarized_docs)
        ]
        return summarize_docs_with_metadata

    def _add_metadata_in_source_docs(self, doc_ids, docs_section, source_docs: list[str]) -> list[Document]:
        """Добавлет metadata в исходные документы: уникальный id документа, принадлежность к группе и позицию документа
        в группе. Сделано для дальнейшей возможности извлечения соседних документов"""
        source_docs_with_metadata = [
            Document(page_content=source, metadata={"doc_id": doc_ids[i], "belongs_to": docs_section, "doc_number": i})
            for i, source in enumerate(source_docs)
        ]
        return source_docs_with_metadata

    def _get_summary_doc_with_metadata(self) -> SummDocsWithSourceAndIds:
        """Возвращает сжатые документы с дополнительными данными, id документов
        и исходные документы
        """
        source_split_documents: list[str] = TextSplitterService.get_semantic_split_documents(self.content)
        summarized_docs: list[Document] = [Document(page_content=sum) for sum in
                                           self._get_summary_doc_content(source_split_documents).summary_texts]
        doc_ids, docs_section = [str(uuid4()) for _ in range(len(summarized_docs))], str(uuid4())

        summarized_docs_with_metadata = self._add_metadata_in_summary_docs(doc_ids, docs_section, summarized_docs)
        source_docs_with_metadata = self._add_metadata_in_source_docs(doc_ids, docs_section, source_split_documents)
        return SummDocsWithSourceAndIds(summarized_docs_with_metadata, doc_ids, source_docs_with_metadata)

    def get_documents_without_add_questions(self, documents: list[Document]) -> list[str]:
        """Удаляет из сжатых текстов дополнительные вопросы, которые были добавлены перед векторизацией"""
        documents_without_questions = [re.sub(r'Вопросы:.*?(?=\n\n|\Z)', '', summ.page_content, flags=re.DOTALL)
                                       for summ in documents]
        return documents_without_questions

    def _define_brief_max_word(self, context: str) -> int:
        len_context = len(context)
        if len_context <= 800:
            return 70
        elif 800 < len_context < 1600:
            return 100
        else:
            return 120

    def super_brief_content(self, documents: list[Document]) -> str:
        documents_content = [doc.page_content for doc in documents]
        context = "\n".join(documents_content)
        return self.model_service.get_super_brief_content(context, self._define_brief_max_word(context))

    def save_docs_and_add_in_retriever(self) -> str:
        """Добавлет документы в векторную базу и возвращает
        краткое содержание без дополнитльно созданных вопросов
        """
        summarize_docs_with_ids, doc_ids, source_docs = self._get_summary_doc_with_metadata()
        user_id = self.retriever.vectorstore._collection_name[5:]
        self.retriever.vectorstore.add_documents(summarize_docs_with_ids)
        DocumentsSaver.save_source_docs_ids_names_in_files(user_id, doc_ids, source_docs)
        DocumentsSaver.add_file_id_with_name_in_file(user_id, source_docs[0].metadata["belongs_to"], self.file_name)
        return self.super_brief_content(source_docs)

    @staticmethod
    def clear_vector_stores(user_id: str):
        """Удаляет векторное хранилище пользователя"""
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            shutil.rmtree(f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
            client.clear_system_cache()
