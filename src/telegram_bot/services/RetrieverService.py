import shutil

import chromadb
import os

from langchain_core.stores import InMemoryStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.telegram_bot.services.custon_multivec_retriever import CustomMultiVecRetriever

os.environ['HF_TOKEN'] = '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'

embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)


class RetrieverSrvice:
    @staticmethod
    def get_or_create_retriever(user_id: str):
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
        id_key = "doc_id"
        print("coll list in get", client.list_collections())
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
        else:
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
        collection_name = f"user_{user_id}"
        client = chromadb.PersistentClient(path=f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
        print("coll list", client.list_collections())
        if collection_name in [name for name in client.list_collections()]:
            shutil.rmtree(f"/home/alex/PycharmProjects/pythonProject/src/chroma_db_{user_id}")
            client.clear_system_cache()
            print("удаление ретривера")

# if __name__ == "__main__":
#     user_id = "12345"
#
#
#     def check_exist_user_directory(user_id: str) -> None:
#         os.makedirs(f'users_directory/user_{user_id}', exist_ok=True)
#
#
#     def save_source_docs_in_files(user_id: str, docs_id: List[str], documents: List[str]) -> None:
#         check_exist_user_directory(user_id)
#         for doc_id, document in zip(docs_id, documents):
#             with open(f'users_directory/user_{user_id}/{doc_id}.txt', 'w') as file:
#                 file.write(document)
#         print("сохранили файлы")
#
#
#     retriever = get_or_create_retriever(user_id)
#     docs = [
#         # Document(
#         #     page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
#         # ),
#         Document(
#             page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
#         ),
#         # Document(
#         #     page_content="Building an exciting new project with LangChain - come check it out!",
#         #
#         # )
#     ]
#
#     uuids = [str(uuid4()) for _ in range(len(docs))]
#     docs_with_ids = [
#         Document(page_content=content.page_content, metadata={"doc_id": uuids[i]}) for i, content in enumerate(docs)
#     ]
#     retriever.vectorstore.add_documents(docs_with_ids)
#     save_source_docs_in_files(user_id, uuids, [d.page_content for d in docs_with_ids])
#     # retriever.docstore.mset(list(zip(uuids, docs)))
#     # print(retriever.docstore.mget([uuids[0]]))
#     print("поиск")
#     print(retriever.invoke("scrambled eggs for breakfast"))
