import shutil
import chromadb
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

os.environ['HF_TOKEN'] = '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'

embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)


class CustomMultiVecRetriever(MultiVectorRetriever):

    @staticmethod
    def __get_source_docs(collection_name: str, doc_id: str) -> List[Document]:
        with open(rf"/home/alex/PycharmProjects/pythonProject/src/users_directory/{collection_name}/{doc_id}.txt",
                  'r') as f:
            content = f.readlines()
            doc = Document(page_content="".join(content))
        return [doc]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) \
            -> list[tuple[Document, float]]:
        result_search_sim_docs = self.vectorstore.similarity_search_with_score(
            query, **self.search_kwargs
        )
        for result_search_sim_doc in result_search_sim_docs:
            collection_name = self.vectorstore._collection_name
            doc_id = result_search_sim_doc[0].metadata["doc_id"]
            source_doc = self.__get_source_docs(collection_name, doc_id)
            result_search_sim_doc[0].metadata["source_doc"] = source_doc

        return result_search_sim_docs


class RetrieverSrvice:
    @staticmethod
    def get_or_create_retriever(user_id: str):
        """Создает векторноую базу и retriever для пользователя, если она не была найдена
        Если такое хранилище существует, возвращает существующие хранилище
        """
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
        id_key = "doc_id"
        store = InMemoryStore()
        if collection_name in [name for name in client.list_collections()]:
            collection = client.get_collection(collection_name)

            vec_store = Chroma(
                collection_name=collection.name,
                embedding_function=embeddings,
                client=client
            )

            retriever = CustomMultiVecRetriever(
                vectorstore=vec_store,
                docstore=store,
                id_key=id_key,
                search_kwargs={"k": 5}
            )

            return retriever
        
        vec_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}",
        )
        retriever = CustomMultiVecRetriever(
            vectorstore=vec_store,
            docstore=store,
            id_key=id_key,
            search_kwargs={"k": 5}
        )

        return retriever

    @staticmethod
    def clear_retriever(user_id: str):
        """Удаляет всю папку пользователя с фргментами текста""""
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
        if collection_name in [name for name in client.list_collections()]:
            shutil.rmtree(f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
            client.clear_system_cache()
