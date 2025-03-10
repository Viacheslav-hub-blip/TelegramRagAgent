from langchain_chroma import Chroma
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_gigachat.chat_models import GigaChat
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langgraph.graph import StateGraph, START, END

from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
from typing_extensions import TypedDict
from typing import List
import logging
from pprint import pprint

from src.custon_multivec_retriever import CustomMultiVecRetriever

logger = logging.getLogger(__name__)

formatter = "%(asctime)s;%(levelname)s;%(message)s"
logging.basicConfig(filename='myapp.log', level=logging.DEBUG, format=formatter)

os.environ['HF_TOKEN'] = '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'
os.environ["TAVILY_API_KEY"] = "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"
os.environ['GROQ_API_KEY'] = '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'
os.environ[
    "GIGACHAT_API_PERS"] = "ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmViOTBhNDZmLTAxNzktNDY4Yi04ODljLTc3ZDZhOTA0YmJjZg=="

model_repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# ----------------------------------Добавление документов------------------------------------------
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
        page_content="Slava Rylkov this is a young developer who is 20 years old. He lives in Moscow, studies at the university. SLava is interested in programming and language models.",
        metadata={"source": "tweet", "doc_id": "2"},
        id=2,
    ),
    Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news", "doc_id": "3"},
        id=3),
    Document(
        page_content="The latest of  SLava project was the development of a financial platform and the creation of a smart assistant.",
        metadata={"source": "tweet", "doc_id": "4"},
        id=4,
    )
]

# uuids = [str(uuid4()) for _ in range(len(docs))]
retriever.vectorstore.add_documents(docs)
retriever.docstore.mset(list(zip(["1", "2", "3"], docs)))

# ----------------------------------Добавление документов------------------------------------------

# ----------------------------------Binary answer LLM---------------------------------------------

llm = GigaChat(verify_ssl_certs=False,
               credentials="ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmViOTBhNDZmLTAxNzktNDY4Yi04ODljLTc3ZDZhOTA0YmJjZg==")
structured_llm_binary = llm

system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

binary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)
retriever_binary_chain = binary_prompt | structured_llm_binary | StrOutputParser()
# ----------------------------------Generate final answer LLM---------------------------------------------

prompt = hub.pull("rlm/rag-prompt")

rag_chain = prompt | RunnablePassthrough(lambda x: logger.warning(f"--FINAL PROMPT-- {x}")) | llm | StrOutputParser()

# ----------------------------------RE WRITE question LLM---------------------------------------------

system_rewrite = """
     You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
"""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        ("human", "Here is the initial question: \n\n {question}")
    ]
)

re_write_question_chain = re_write_prompt | llm | StrOutputParser()

# ----------------------------------Web search tool---------------------------------------------


web_search_tool = TavilySearchResults(k=3)


# ----------------------------------Create Graph---------------------------------------------
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    web_searched_docs: List[Document]


def retrieve(state: GraphState):
    """
       Retrieve documents

       Args:
           state (dict): The current graph state

       Returns:
           state (dict): New key added to state, documents, that contains retrieved documents
       """
    logger.info("--RETRIEVE--")
    question = state["question"]
    documents = retriever.invoke(question)
    logger.info(f"--RETRIEVE--RETRIEVED DOCS-- \n {documents}")
    return {"documents": documents, "question": question}


def binary_classification(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    logger.info("--CHECK DOCUMENT RELEVANCE TO QUESTION--")
    question = state["question"]
    documents = state["documents"]

    web_search = "No"
    if len(documents) == 0:
        web_search = "Yes"
        logger.warning("EMPTY DOCUMENTS----WEB SEARCH")
    else:
        documents_content = "\n".join([doc[0].page_content for doc in documents])
        binary_res = retriever_binary_chain.invoke({"question": question, "document": documents_content})
        logger.debug(f"BINARY RES {binary_res}")
        if binary_res in "yes" or binary_res in "YES" or binary_res == 'Yes':
            logger.info("-BINARY SEARCH YES--")
        else:
            web_search = "Yes"
            logger.info("--BINARY SEARCH NO,  WEB SERACH--")
    return {"question": question, "documents": documents, "web_search": web_search}


def transform_query(state: GraphState):
    """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
    """

    logger.info("--TRANSFORM QUERY--")
    question = state["question"]
    documents = state["documents"]

    better_question = question
    logger.debug(f"BETTER QUESTION: {better_question}")
    return {"documents": documents, "question": better_question}


def web_search(state: GraphState):
    """
       Web search based on the re-phrased question.

       Args:
           state (dict): The current graph state

       Returns:
           state (dict): Updates documents key with appended web results
       """

    logger.info("--WEB SEARCH--")
    question = state["question"]
    logger.debug(f"WEB SEARCH QUESTION {question}")
    docs_search = web_search_tool.invoke({"query": question})
    logger.debug(f"DOCS SEARCHED Find {docs_search}")
    web_results = "\n".join([d["content"] for d in docs_search])
    web_results = [Document(page_content=web_results)]

    return {"web_searched_docs": web_results, "question": question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    logger.info("--ASSESS GRADED DOCUMENTS--")
    web_search_state = state["web_search"]

    if web_search_state == "Yes":
        logger.warning("-DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY--")
        return "transform_query"
    else:
        logger.info("-DECISION: GENERATE--")
        return "generate"


def generate(state: GraphState):
    """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

    logger.info("--GENERATE--")
    question = state["question"]
    web_search_state = state["web_search"]

    if web_search_state == "Yes":
        documents = state["web_searched_docs"]
        documents_content = "\n".join([doc.page_content for doc in documents])
    else:
        documents = state["documents"]
        documents_content = "\n".join([doc[0].page_content for doc in documents])
    logger.warning(f"--FINAL DOCS-- {documents}")

    generation = rag_chain.invoke({"context": documents_content, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# ----------------------------------COMPILE Graph---------------------------------------------


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", binary_classification)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_tool", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
workflow.add_edge("transform_query", "web_search_tool")
workflow.add_edge("web_search_tool", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ----------------------------------RUN Graph---------------------------------------------

inputs = {"question": "who is Slava Rylkov and what does he do?"}
result = app.invoke(inputs)
question, generation, web_search = result["question"], result["generation"], result["web_search"]
documents: List[Document] = result["documents"]
print("##QUESTION## ", question)
print("##ANSWER## ", generation)
print("##WEB_SERACH## ", web_search)
pprint(documents)
