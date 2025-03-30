import logging
from typing import List, TypedDict, Literal
from langchain_core.tools import BaseTool
from langchain import hub
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
    question_category: str
    question_with_additions: str


class RagAgent:
    def __init__(self, model: BaseChatModel, retriever, web_search_tool: BaseTool):
        self.model = model
        self.retriever = retriever
        self.web_search_tool = web_search_tool
        self.state = GraphState

    def simple_chain(self, system_prompt: str, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt)
            ]
        )

        answer_chain = prompt | self.model | StrOutputParser()
        return answer_chain.invoke({"question": question})

    def analyze_query_for_category_chain(self, question: str) -> str:
        system_prompt = """
                Ты умный ассистент, который разделяет запрс пользователя на 3 категории:\n
                1.Фактическая - если в вопросе спрашивают про какие либо факты, либо если они должны содержаться в ответе\n
                2.Аналитическая  - если для ответа на вопрос нужно провести цепочку рассуждений \n
                3.Мнение - если в вопросе спрашивают твое (языковой модели) мнение или просят порассуждать\n

                Отнеси вопрос к одной из категорий и ответь одним словом: Factual, Analytical, Opinion\n
                Вопрос пользователя: {question}
                """

        return self.simple_chain(system_prompt, question)

    def analyze_query_for_category(self, state: GraphState):
        """Анализирует вопрос и разделяет его на 3 категории:
        1. Фактическая
        2. Аналитическая
        3. Формирование мнения
        В зависимости от категории на следующих этапах убудут сформированы вспомогательные вопросы
        """
        question_category: str = self.analyze_query_for_category_chain(state["question"])
        print("question_category", question_category)
        return {"question": state["question"], "question_category": question_category}

    def choose_query_strategy(self, state: GraphState):
        """Взависимости от выбранной категории вопроса перенаправляет в цепочку"""
        query_strategy: str = state["question_category"].lower()

        if query_strategy == "factual":
            return "Factual"
        elif query_strategy == "analytical":
            return "Analytical"
        return "Opinion"

    def factual_query_chain(self, question: str) -> str:
        system_prompt = """
                    Вы умный помощник, который помогает улучшить или дполнить вопрос пользователя для дальнейшего\n
                    извлечения информации по этому вопросу\n
                    Ваша задача: Улучшите этот фактологический запрос для лучшего поиска информации\n
                    Вопрос пользователя: {question}
                """
        return self.simple_chain(system_prompt, question)

    def factual_query_strategy(self, state: GraphState):
        """Цепочка, которая выполняется если выбран тип вопроса 'Фактический'
        В этом случае генерируются дполнительные воросы
        """
        question_with_additions: str = self.factual_query_chain(state["question"])
        return {"question_with_additions": question_with_additions}

    def analytical_query_chain(self, question: str) -> str:
        system_prompt = """
                    Вы умный помощник, который помогает улучшить или дполнить вопрос пользователя для дальнейшего\n
                    извлечения информации по этому вопросу\n
                    Ваша задача: Улучшите этот аналитический запрос для лучшего поиска информации\n
                    Например, добавьте несколько уточняющих вопросов.\n
                    
                    Пример работы:\n
                    Тестовый вопрос пользователя: Сколько гениев в городе Нью-йорк?\n
                    Дополнительные вопросы:\n
                    Кого можно назвать гением?\n
                    Какое население в этом городе?\n
                    Как часто встречаются гении?\n
                    
                    Пример вашего ответа: 
                    Сколько гениев в городе Нью-йорк?\n
                    Кого можно назвать гением?\n
                    Какое население в этом городе?\n
                    Как часто встречаются гении?\n
                    
                    Вопрос пользователя: {question}
                    """
        return self.simple_chain(system_prompt, question)

    def analytical_query_strategy(self, state: GraphState):
        """Цепочка которая выполняется в случае если выбран тип вопроса 'Аналитический'
        Для такого вопроса генерируются уточняющие вопросы
        """
        question_with_additions: str = self.analytical_query_chain(state["question"])
        return {"question_with_additions": question_with_additions}

    def opinion_query_chain(self, question: str) -> str:
        system_prompt = """
                    Вы умный помощник, который помогает улучшить или дполнить вопрос пользователя для дальнейшего\n
                    извлечения информации по этому вопросу\n
                    Ваша задача: Улучшите этот вопрос про ваше мнение, добавив в него вопросы про различные точки зрения\n
        """
        return self.simple_chain(system_prompt, question)

    def opinion_query_strategy(self, state: GraphState):
        """Цепочка которая выполняется в случае если выбран тип вопроса 'Формирование мнения'
        Для такого вопроса генерируются уточняющие вопросы
        """
        question_with_additions: str = self.opinion_query_chain(state["question"])
        return {"question_with_additions": question_with_additions}

    def retrieve_documents(self, state: GraphState):
        searched_documents: List[Document] = self.retriever.get_relevant_documents(state["question"])

