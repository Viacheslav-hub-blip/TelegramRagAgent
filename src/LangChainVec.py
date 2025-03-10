from uuid import uuid4
from langchain.schema.document import Document
import os

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from src.custon_multivec_retriever import CustomMultiVecRetriever
from transformers import AutoTokenizer

os.environ['HF_TOKEN'] = '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'

# embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1", task="feature-extraction")


retriever = None
added = False


def get_retriever():
    global retriever
    if retriever is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1"
        )
        vec_store = Chroma(
            collection_name="RagVecStore",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )

        store_doc = InMemoryStore()
        id_key = "doc_id"

        retriever = CustomMultiVecRetriever(
            vectorstore=vec_store,
            docstore=store_doc,
            id_key=id_key,
            search_kwargs={"k": 5}
        )
        print("!!!!!!!Создали CustomMultiVecRetriever!!!!!!")
        return retriever
    else:
        print("!!!!!!!Вернули CustomMultiVecRetriever!!!!!!")
        return retriever


if __name__ == "__main__":
    docs = [
        Document(
            page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
            metadata={"source": "tweet"},
            id=1,
        ),
        Document(
            page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
            metadata={"source": "news"},
            id=2),
        Document(
            page_content="Building an exciting new project with LangChain - come check it out!",
            metadata={"source": "tweet"},
            id=3,
        )
    ]

    uuids = [str(uuid4()) for _ in range(len(docs))]
    retriever.vectorstore.add_documents(docs)
    retriever.docstore.mset(list(zip(uuids, docs)))
    print(retriever.docstore.mget([uuids[0]]))
    print(retriever.vectorstore.similarity_search_with_score("scrambled eggs for breakfast"))
