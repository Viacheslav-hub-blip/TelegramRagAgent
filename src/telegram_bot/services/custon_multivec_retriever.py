from pprint import pprint
from typing import List, Tuple
from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from uuid import uuid4
from langchain.schema.document import Document
import os

from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.retrievers.multi_vector import MultiVectorRetriever


class CustomMultiVecRetriever(MultiVectorRetriever):

    @staticmethod
    def __get_source_docs(collection_name: str, doc_id: str) -> List[Document]:
        # return self.docstore.mget([doc_id])
        with open(rf"users_directory/{collection_name}/{doc_id}.txt", 'r') as f:
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

            # source_doc = self.docstore.mget([doc_id])

            source_doc = self.__get_source_docs(collection_name, doc_id)
            result_search_sim_doc[0].metadata["source_doc"] = source_doc

        return result_search_sim_docs


if __name__ == '__main__':
    os.environ['HF_TOKEN'] = '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'

    embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1", task="feature-extraction")
    vec_store = Chroma(
        collection_name="example",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    #
    # vec_store = InMemoryVectorStore(embedding=embeddings)

    store_doc = InMemoryStore()
    id_key = "doc_id"

    retriever = CustomMultiVecRetriever(
        vectorstore=vec_store,
        docstore=store_doc,
        id_key=id_key,
        search_type="mmr",
        search_kwargs={"k": 2}
    )

    docs = [
        Document(
            page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
            metadata={"source": "tweet", "doc_id": "1"},
            id=1,
        ),
        Document(
            page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
            metadata={"source": "news", "doc_id": "2"},
            id=2),
        Document(
            page_content="Building an exciting new project with LangChain - come check it out!",
            metadata={"source": "tweet", "doc_id": "3"},
            id=3,
        )
    ]

    # uuids = [str(uuid4()) for _ in range(len(docs))]
    retriever.vectorstore.add_documents(docs, collection_metadata={"hnsw:space": "cosine"})
    retriever.docstore.mset(list(zip(["1", "2", "3"], docs)))

    query = "LangChain"
    pprint(retriever.invoke(query))
