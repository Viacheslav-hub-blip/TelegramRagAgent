import chromadb
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.telegram_bot.embedding import embeddings
from src.telegram_bot.services.documents_getter_service import DocumentsGetterService
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class CustomRetriever:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    def get_relevant_documents(self, query: str, belongs_to: str = None) -> list[Document]:
        if belongs_to:
            result_search_sim_docs = self.vectorstore.similarity_search_with_score(query, k=10,
                                                                                   filter={"belongs_to": belongs_to})
        else:
            result_search_sim_docs = self.vectorstore.similarity_search_with_score(query)
        # print(result_search_sim_docs)
        collection_name = self.vectorstore._collection_name
        result = []
        for result_search_sim_doc, score in result_search_sim_docs:
            doc_id = result_search_sim_doc.metadata["doc_id"]
            belongs_to = result_search_sim_doc.metadata["belongs_to"]
            doc_number = result_search_sim_doc.metadata["doc_number"]
            result_search_sim_doc.metadata["score"] = score
            source_doc = DocumentsGetterService.get_source_document(collection_name, doc_id, belongs_to, doc_number)
            result_search_sim_doc.metadata["source_doc"] = source_doc.page_content
            result.append(result_search_sim_doc)
        # for r in result:
        #     print(r.metadata["score"], r.metadata["source_doc"])
        # print("result search result", result)
        return result


class RetrieverSrvice:

    @staticmethod
    def get_or_create_retriever(user_id: str):
        """Создает векторноую базу и retriever для пользователя, если она не была найдена
        Если такое хранилище существует, возвращает существующие хранилище
        """
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            collection = client.get_collection(collection_name)
            vec_store = Chroma(
                collection_name=collection.name,
                embedding_function=embeddings,
                client=client,
                collection_metadata={"hnsw:space": "cosine"},

            )
            retriever = CustomRetriever(
                vectorstore=vec_store,
            )
            return retriever

        vec_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}",
            collection_metadata={"hnsw:space": "cosine"}
        )
        retriever = CustomRetriever(
            vectorstore=vec_store,
        )
        return retriever
