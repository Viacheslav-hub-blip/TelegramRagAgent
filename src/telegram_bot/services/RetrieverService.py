import chromadb
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List, Any
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.telegram_bot.config import HF_TOKEN, embeddings_model_name
from langchain_core.retrievers import BaseRetriever

os.environ['HF_TOKEN'] = HF_TOKEN

embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_name
)


class CustomRetriever:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    @staticmethod
    def __get_source_docs(collection_name: str, doc_id: str, belongs_to: str, doc_number: str) -> List[Document]:
        with open(
                rf"/home/alex/PycharmProjects/pythonProject/src/users_directory/{collection_name}/{doc_id}/{belongs_to}/{doc_number}.txt",
                'r') as f:
            content = f.readlines()
            doc = Document(page_content="".join(content))
        return [doc]

    def get_relevant_documents(self, query: str) -> list[Document]:
        result_search_sim_docs = self.vectorstore.similarity_search_with_score(query)
        collection_name = self.vectorstore._collection_name
        result = []
        for result_search_sim_doc, score in result_search_sim_docs:
            doc_id = result_search_sim_doc.metadata["doc_id"]
            belongs_to = result_search_sim_doc.metadata["belongs_to"]
            doc_number = result_search_sim_doc.metadata["doc_number"]
            source_doc = self.__get_source_docs(collection_name, doc_id, belongs_to, doc_number)
            result_search_sim_doc.metadata["source_doc"] = source_doc
            result_search_sim_doc.metadata["score"] = score
            result.append(result_search_sim_doc)
        print("result search result")
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
                collection_metadata={"hnsw:space": "cosine"}
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
