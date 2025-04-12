import logging
from typing import List, TypedDict
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
import warnings


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
    forced_generation: str
    documents: List[Document]
    web_searched_docs: List[Document]


class RagAgent:
    def __init__(self, model: BaseChatModel, retriever, web_search_tool: BaseTool):
        self.model = model
        self.retriever = retriever
        self.logger = logging.getLogger(self.__class__.__name__)
        self.web_search_tool = web_search_tool
        self.state = GraphState
        self.app = self.__compile_graph()

    def _retriever_binary_answer_chain(self, document: str, question: str) -> str:
        system = """
            Вы оцениваете соответствие полученного документа вопросу пользователя. \n
            Если документ по смыслу связан с вопросом или содержит ключевые слова или по нему можно ответить на вопрос\n
            и может быть ответом на вопрос, оцените его как "ДА". \n
            Присвойте двоичную оценку "да" или "нет", чтобы указать, соответствует ли документ вопросу.\n
            ИСПОЛЬЗУЙТЕ ТОЛЬКО "ДА" или "НЕТ".\n
            Если вопрос задан на другую тему или про других персонажей или по контексту нельзя дать ответ, ответь "Нет".\n
                
            Например, если в вопросе спрашивают про  элемент "EL1" и он содержится в контексте, ответьте "ДА". Если "EL1" \n
            не указан, но косвенно упоминается, ответьте "ДА".Если по контексту можно дать ответ, ответьте "Да".\n
        """

        binary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
            ]
        )
        binary_chain = binary_prompt | self.model | StrOutputParser()
        return binary_chain.invoke({"document": document, "question": question})

    def _rag_answer_chain(self, context: str, question: str) -> str:
        system = """
                 You are an assistant for question-answering tasks. Use the following content and question.\n
                 If the context is empty and you don't know the answer, just say you don't know. \n
                 Answer the question as fully and accurately as possible.\n
                 Context: {context}
                 Question: {question}
                 """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system)
            ]
        )
        print('RAG PROMPT', prompt)

        rag_chain = prompt | RunnablePassthrough(
            lambda x: self.logger.warning(f"--FINAL PROMPT-- {x}")) | self.model | StrOutputParser()
        return rag_chain.invoke({"context": context, "question": question})

    def _re_write_question_chain(self, question: str) -> str:
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

        re_write_question_chain = re_write_prompt | self.model | StrOutputParser()
        return re_write_question_chain.invoke({"question": question})

    def _usage_web_search_chain(self, question: str) -> str:
        system_prompt = """
        Ты умный помощник, который определяет необходимо ли использовать поиск в интернета или можно использовать\n
        собсвенные знания. \n

        Например, использовать посик в интернете необходимо когда: пользователь спрашивает по текущее события или события, которые \n
        произошли в последнее время или когда ответ должен содеражть какую то историческую или точную ифнормацию.\n
        Например, пользователь спрашивает про погоду/новости/последние события/известных личностей или про вещи, которые могли изменяться со времнем.\n

        Собственные знания необходимо использовать, когда пользователь просит написать какой-либо текст, например, сочинение или поздравление \n
        а также когда ответ не должен содержать каких либо точных данных.\n

        Твоя задача: проанализируй вопрос и определи нужно ли использовать поиск. Твой ответ должен состоять из одного слова "Да" или "Нет".\n
        Не используй в ответе более одногос слова

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt)
            ]
        )
        usage_web_chain = prompt | self.model | StrOutputParser()
        return usage_web_chain.invoke({"question": question})

    def _answer_on_own_knowledge_chain(self, question: str) -> str:
        system_prompt = """
        При обучении языковой модели важно не только количество данных, но и их качество.\n
        Давайте сосредоточимся на том, чтобы улучшить качество генерации текста, делая его более разнообразным, точным и осмысленным. \n
        При создании новых предложений, пожалуйста, учитывайте контекст, грамматику, стилистическое разнообразие и семантическую связность.\n
        если модель начинает повторяться или выдавать неуместные ответы, дайте ей подсказку или переформулируйте запрос таким образом, \n
        чтобы она могла лучше понять задачу.\n
        Размышляй шаг за шагом и старайтся дать наиболее правильный ответ.\n    
        НЕ нужно печатать все твои размышление. В результате выведи только конечный ответ.\n    
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt)
            ]
        )
        answer_chain = prompt | self.model | StrOutputParser()
        return answer_chain.invoke({"question": question})

    def _route_web_search_or_generate(self, state: GraphState):
        if state["web_search"] == "Yes":
            return "transform_query"
        return "generate"

    def _decide_to_forced_generation(self, state: GraphState):
        if state["forced_generation"] == "YES":
            return "generate"
        else:
            return "grade_documents"

    def near_documents(self, target_doc_id: str, target_doc_section: str, k: int = 1) -> List[Document]:
        pass

    def retrieve_doc_search(self, state: GraphState):
        """
               Retrieve documents

               Args:
                   state (dict): The current graph state

               Returns:
                   state (dict): New key added to state, documents, that contains retrieved documents
        """
        self.logger.info("--RETRIEVE--")
        question: str = state["question"]
        try:
            documents: List[Document] = self.retriever.get_relevant_documents(question)
        except Exception as e:
            print('Ошибка извлечения документов', Exception, e)
            exit()

        if len(documents) == 0:
            return {"forced_generation": "YES", "question": question}
        self.logger.warning(f"RETRIEVED DOCS-- \n {len(documents)} \n {documents}")
        return {"documents": documents, "question": question, "forced_generation": "NO"}

    def binary_classification(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        self.logger.info("--CHECK DOCUMENT RELEVANCE TO QUESTION--")
        question = state["question"]
        documents = state["documents"]

        web_search = "No"
        if len(documents) == 0:
            web_search = "Yes"
            self.logger.warning("EMPTY DOCUMENTS----WEB SEARCH")
        else:
            documents_content = "\n".join([doc.page_content for doc in documents])
            binary_res = self._retriever_binary_answer_chain(documents_content, question)
            self.logger.warning(f"BINARY CLASSIFICASION RES {binary_res}")
            if binary_res in ["yes", "YES", "Yes", "Да", "ДА"]:
                self.logger.info("-BINARY SEARCH YES--")
            else:
                web_search = "Yes"
                self.logger.info("--BINARY SEARCH NO,  WEB SERACH--")
        return {"question": question, "documents": documents, "web_search": web_search}

    def transform_query(self, state: GraphState):
        """
            Transform the query to produce a better question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates question key with a re-phrased question
        """

        self.logger.info("--TRANSFORM QUERY--")
        question = state["question"]

        better_question = question
        self.logger.debug(f"BETTER QUESTION: {better_question}")
        return {"question": better_question}

    def web_search(self, state: GraphState):
        """
           Web search based on the re-phrased question.

           Args:
               state (dict): The current graph state

           Returns:
               state (dict): Updates documents key with appended web results
           """
        self.logger.info("--WEB SEARCH--")
        question = state["question"]
        self.logger.debug(f"WEB SEARCH QUESTION {question}")
        docs_search = self.web_search_tool.invoke({"query": question})
        self.logger.debug(f"DOCS SEARCHED Find {docs_search}")
        web_results = "\n".join([d["content"] for d in docs_search])
        source_links = [doc["url"] for doc in docs_search]
        web_results = [Document(page_content=web_results, metadata={"source": source_links})]

        return {"web_searched_docs": web_results, "question": question}

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        self.logger.info("--ASSESS GRADED DOCUMENTS--")
        web_search_state = state["web_search"]

        if web_search_state == "Yes":
            self.logger.warning("-DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY--")
            return "docs_not_relevant"
        else:
            self.logger.info("-DECISION: GENERATE--")
            return "generate"

    def decide_to_use_web_search(self, state: GraphState):
        question = state["question"]
        usage_web = self._usage_web_search_chain(question)

        if usage_web in ["Да", "ДА", "Yes", "YES"]:
            return {"web_search": "Yes"}
        return {"web_search": "No"}

    def generate_based_on_own_knowledge(self, state: GraphState):
        answer = self._answer_on_own_knowledge_chain(state["question"])
        return {"generation": answer}

    def generate(self, state: GraphState):
        """
            Generate answer

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """

        self.logger.info("--GENERATE--")
        question = state["question"]
        web_search_state = state["web_search"]

        if web_search_state == "Yes":
            documents = state["web_searched_docs"]
            documents_content = "\n".join([doc.page_content for doc in documents])
        else:
            documents = state["documents"]
            documents_content = "\n".join([doc.metadata["source_doc"] for doc in documents])
        self.logger.warning(f"--FINAL DOCS-- {documents}")

        generation = self._rag_answer_chain(documents_content, question)
        return {"documents": documents, "question": question, "generation": generation}

    def __compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("retrieve", self.retrieve_doc_search)
        workflow.add_node("decide_to_use_web_search", self.decide_to_use_web_search)
        workflow.add_node("grade_documents", self.binary_classification)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("web_search_tool", self.web_search)
        workflow.add_node("generate_based_on_own_knowledge", self.generate_based_on_own_knowledge)

        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._decide_to_forced_generation,
            {
                "generate": "decide_to_use_web_search",
                "grade_documents": "grade_documents"
            }
        )
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "docs_not_relevant": "decide_to_use_web_search",
                "generate": "generate",
            }
        )
        workflow.add_conditional_edges(
            "decide_to_use_web_search",
            self._route_web_search_or_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate_based_on_own_knowledge"
            }
        )

        workflow.add_edge("transform_query", "web_search_tool")
        workflow.add_edge("web_search_tool", "generate")
        workflow.add_edge("generate", END)
        workflow.add_edge("generate_based_on_own_knowledge", END)

        return workflow.compile()

    def __call__(self, *args, **kwargs):
        return self.app


if __name__ == "__main__":
    from src.telegram_bot.services.retriever_service import CustomRetriever, embeddings
    from langchain_gigachat.chat_models import GigaChat
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_chroma import Chroma
    import os
    from pprint import pprint
    from src.telegram_bot.config import *

    warnings.filterwarnings("ignore")

    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    os.environ[
        "GIGACHAT_API_PERS"] = GIGACHAT_API_PERS

    vec_store = Chroma(
        collection_name="example",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = CustomRetriever(
        vec_store,
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
        ),
        Document(
            page_content="Слава Рыльков - молодой разработчик, ему 20 лет. Он живет в Москве, учится в университете. Слава интересуется программированием и языковыми моделями.",
            metadata={"source": "tweet", "doc_id": "5"},
            id=5,
        ),
    ]

    # uuids = [str(uuid4()) for _ in range(len(docs))]
    retriever.vectorstore.add_documents(docs)

    llm = GigaChat(verify_ssl_certs=False,
                   credentials=GIGACHAT_API_PERS)

    tool = TavilySearchResults(k=3)

    rag_agent = RagAgent(model=llm, retriever=retriever, web_search_tool=tool)

    while True:
        input_question = input("Введите сообщение: ")

        if input_question != "q":
            inputs = {"question": input_question}
            result = rag_agent().invoke(inputs)
            print(result, result["forced_generation"])
            question, generation, web_search, forced_generation = result["question"], result["generation"], result[
                "web_search"], result["forced_generation"]
            try:
                documents: List[Document] = result["documents"]
            except:
                documents = []

            print("##QUESTION## ", question)
            print("##ANSWER## ", generation)
            print("##WEB_SERACH## ", web_search)
            pprint(documents)
            print()
        else:
            exit()
