from typing import List, TypedDict, Optional

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.telegram_bot.services.documents_getter_service import DocumentsGetterService


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
    file_metadata_id: str
    user_id: str
    question_category: str
    question_with_additions: str
    retrieved_documents: List[Document]
    neighboring_docs: List[Document]
    answer_with_retrieve: str
    answer: str
    answer_without_retrieve: bool
    used_docs: List[str]


class RagAgent:
    def __init__(self, model: BaseChatModel, retriever):
        self.model = model
        self.retriever = retriever
        self.state = GraphState
        self.app = self.compile_graph()

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
                Ты умный ассистент, который разделяет запрос пользователя на 3 категории для улучшения поиска документов в базе:\n
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
                    Ваша задача: Улучшите этот фактологический запрос дополнительными вопросами для лучшего поиска информации. В качестве ответа предоставьте только улучшенный запрос.\n
                    НЕ используй вступительный слова по типу: Улучшенный запрос, новый запрос и т.д
                    Вопрос пользователя: {question}
                """
        return self.simple_chain(system_prompt, question)

    def factual_query_strategy(self, state: GraphState):
        """Цепочка, которая выполняется если выбран тип вопроса 'Фактический'
        В этом случае генерируются дполнительные воросы
        """
        question_with_additions: str = self.factual_query_chain(state["question"])
        print("----factual strategy------", question_with_additions)
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
        print("----analytical strategy------", question_with_additions)
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
        print("----opinion strategy------", question_with_additions)
        return {"question_with_additions": question_with_additions}

    def retrieve_documents(self, state: GraphState):
        """Ищет документы и ограничивает выборку документами со сходством <= 1.3(наиболее релевантные)"""
        retrieved_documents: List[Document] = self.retriever.get_relevant_documents(state["question_with_additions"],
                                                                                    state["file_metadata_id"])
        for d in retrieved_documents:
            print(d)
        print("------retrieve_documents------", retrieved_documents)
        return {"retrieved_documents": retrieved_documents}

    def get_neighboring_numbers_doc(self, section_numbers_dict: dict) -> dict:
        """Получает словарь, где ключ - раздел документа, значение - номера документов в разделе
        Возвращает словарь, где к номерам документов добавляютс соседние номера кажого документа
        """
        res_dict = {}
        for sec, numbers in section_numbers_dict.items():
            numbers_int: list[int] = [int(s) for s in numbers.split("/")]
            print("numbers_int", numbers_int)
            neighboring_numbers: list[int] = numbers_int + [n + 1 for n in numbers_int]
            print("neighboring_numbers", neighboring_numbers)
            unique_neighboring_numbers = sorted(set(neighboring_numbers))
            print("unique_neighboring_numbers", unique_neighboring_numbers)
            res_dict[sec] = "/".join([str(i) for i in unique_neighboring_numbers])
        print("res_dict", res_dict)
        return res_dict

    def get_neighboring_docs(self, state: GraphState):
        """Ищет соседние исходные документы к тем, что были надйены при посике с помощью retriever"""
        section_numbers_dict = {}
        for doc in state["retrieved_documents"]:
            if doc.metadata["belongs_to"] in section_numbers_dict:
                section_numbers_dict[doc.metadata["belongs_to"]] += f'/{doc.metadata["doc_number"]}'
            else:
                section_numbers_dict[doc.metadata["belongs_to"]] = str(doc.metadata["doc_number"])
        neighboring_docs_numbers: dict = self.get_neighboring_numbers_doc(section_numbers_dict)
        print("neighboring_docs_numbers", neighboring_docs_numbers)
        neighboring_docs: list[Document] = []
        for sec, numbers in neighboring_docs_numbers.items():
            doc_nums = numbers.split("/")
            for num in doc_nums:
                document = DocumentsGetterService.get_document_by_user_id_section_and_number(state["user_id"], sec, num)
                neighboring_docs.append(document)
        print("count docs", len(neighboring_docs))
        print("------get_neighboring_docs------", neighboring_docs)
        return {"neighboring_docs": neighboring_docs}

    def checking_possibility_responses_chain(self, question: str, context: str):
        """Просим модель проверить, можно ли дать ответ на вопрос по найденным документам"""
        prompt = """
        Ты - умный помощник, который должен определить, можно ли ответить на вопрос пользователя по найденному контексту.
        Проанализируй весь полученный контекст, выясни, есть ли в контексте ключевые слова из вопроса, подходит ли контекст к вопросу по семантике и смыслу.
        Всеми силами попробуй дать ответ на вопрос пользователя по найденному контексту.
        Если по найденному контексту можно дать ответ на вопрос пользователя или контекст содержит ключевые слова которые есть в вопросе или похож по семантике, напиши "ДА". 
        Если по найденному конкесту нельзя дать ответ на вопрос пользователя, то ответь "НЕТ".Отвечать "НЕТ" необходимо только в том случае, когда контекст полностью не соотвествует вопросу.
        Не используй другие слова в ответе.
        
        Вопрос пользователя:{question}
        
        Найденный контекст: {context}
        """

        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt)
            ]
        )

        chain = system_prompt | self.model | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    def checking_possibility_responses(self, state: GraphState):
        """Получает результат оценки возможности ответа на вопрос по контексту"""
        doc_context = "".join([doc.page_content for doc in state["neighboring_docs"]])
        binary_check = self.checking_possibility_responses_chain(state["question"], doc_context)
        print("------checking_possibility_responses------", binary_check)
        if binary_check.lower() in ["yes", "да"]:
            return "да"
        else:
            return "нет"

    def answer_with_context_chain(self, question: str, context: str):
        prompt = """
        Ты — интеллектуальный ассистент, который анализирует предоставленный контекст и формулирует точный, информативный ответ на вопрос пользователя.\n
        Инстуркции:\n
        Внимательно прочитай контекст и вопрос.\n
        
        Ответ должен быть:\n
        -Ракрытым, грамотным и содержащим полную информацию\n
        -Каждый ответ должен быть точным, правдимым и отвечать на вопрос пользователя\n
        -Основанным только на контексте (не добавляй внешних знаний).\n
        
        Для этого тебе стоит:\n
        -Рассуждать шаг за шагом\n
        -Продумывать свои ответы и действия\n
        -Проверять правильность построения предложений\n
        -Использовать разные части речи\n
        
        
        Вопрос: {question}\n
        
        Найденный контекст: {context}\n
        
        Важно: проверь правильность всех фактов, которые пишешь. Проверь имена действующих лиц и соотвествие твоего ответа реальности.
        Не описывай все предыщие шаги о размышлениях в ответе. Дай только ответ
        """
        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt)
            ]
        )
        chain = system_prompt | RunnablePassthrough(
            lambda x: print('Final prompt:\n', x)) | self.model | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    def generate_answer_with_retrieve_context(self, state: GraphState):
        doc_context = "".join([doc.page_content for doc in state["neighboring_docs"]])
        answer = self.answer_with_context_chain(state["question"], doc_context)
        print("------generate_answer_with_retrieve_context------", answer)
        return {"answer_with_retrieve": answer}

    def answer_without_context_chain(self, question: str):
        prompt = """
                Ты — интеллектуальный ассистент, который анализирует вопрос и формулирует точный, информативный ответ на вопрос пользователя.        
                Инстуркции:
                Ответ должен быть:
                -Кратким и по делу (если не указано иное).
                -Структурированным (используй списки, абзацы, выделение ключевых моментов при необходимости).

                Вопрос: {question}
                """
        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt)
            ]
        )
        chain = system_prompt | self.model | StrOutputParser()
        return chain.invoke({"question": question})

    def answer_without_context(self, state: GraphState):
        answer = self.answer_without_context_chain(state["question"])
        print("------answer_without_context------", answer)
        return {"answer_without_retrieve": True, "answer": answer}

    def check_answer_for_correctness_chain(self, retrieve_answer: str, question: str):
        prompt = """
                   Ты - эксперт-редактор, проверяющий ответы, сгенерированные на основе найденных документов. Твоя задача - тщательно анализировать предоставленный ответ в контексте исходного вопроса пользователя и имеющихся данных, затем либо подтверждать его корректность, либо вносить необходимые исправления.

                    Алгоритм работы:
                    1. Внимательно проанализируй вопрос пользователя и предоставленный ответ
                    2. Проверь ответ по следующим критериям:
                       - Соответствие вопросу
                       - Фактическая точность (все утверждения должны быть подтверждены документами)
                       - Логическая целостность (отсутствие противоречий в ответе)
                       - Полнота (ответ должен охватывать ключевые аспекты вопроса)
                       - Ясность изложения (информация должна быть понятной и структурированной)
                    
                    3. Если ответ полностью удовлетворяет всем критериям:
                       - Верни исходный текст без изменений
                    
                    4. Если найдены ошибки или неточности:
                       - Предоставь исправленную версию ответа
                       - Сохрани исходный смысл, но сделай текст более точным и соответствующим вопросу
                    
                    5. Особое внимание удели:
                        -Названиям,именам,действующим лицам и персонажам
                       - Недопущению галлюцинаций (вымышленных фактов)
                       - Корректной интерпретации данных из документов
                       - Сохранению нейтрального тона и объективности
                    
                    6.Не удаляй факты,которые были найдены при поиске
                    
                    7. ЕСЛИ ИЗМЕНЕНИЯ НЕ ТРЕБУЮТСЯ ОТСАВЬ ИСХОДНЫЙ ТЕКСТ(retrieve_answer) БЕЗ ИЗМЕНЕНИЙ
                    
                    Пример работы:                    
                    Вопрос: "Когда была подписана Декларация независимости США?"
                    Ответ: "Американская декларация была подписана в 1876 году."
                    Исправленный ответ: "Декларация независимости США была подписана 4 июля 1776 года."
                    
                    Теперь проанализируй следующий вопрос и ответ:
                    
                    Вопрос: {question}
                    Ответ: {retrieve_answer}
                    
                    ЕСЛИ ИЗМЕНЕНИЯ НЕ ТРЕБУЮТСЯ НАПИШИ ИСХОДНЫЙ ТЕКСТ(retrieve_answer) БЕЗ ИЗМЕНЕНИЙ
                    Минимум дополнений: Добавляй информацию только тогда, когда это критично для точности или полноты. Если ответ соотвествует вопросу, то НЕ ДОБАВЛЯЙ НИЧЕГО.\n
                    В качестве ответа напиши только итоговый ответ без описания всех предыдищх шагов\n
                    """

        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt)
            ]
        )

        chain = system_prompt | self.model | StrOutputParser()
        return chain.invoke(
            {"retrieve_answer": retrieve_answer, "question": question})

    def _delete_special_symbols(self, answer: str) -> str:
        answer = answer.replace("**", "")
        return answer

    def check_answer_for_correctness(self, state: GraphState):
        retrieve_answer = state["answer_with_retrieve"]
        #final_answer = self.check_answer_for_correctness_chain(retrieve_answer, state["question"])
        final_answer = self._delete_special_symbols(retrieve_answer)
        print("------check_answer_for_correctness------", final_answer)
        return {"answer_without_retrieve": False, "answer": final_answer}

    def add_source_docs_names(self, state: GraphState):
        documents: list[Document] = state["retrieved_documents"]
        file_ids_names = DocumentsGetterService.get_files_ids_names(state["user_id"])
        used_docs_names = list(set([file_ids_names.get(doc.metadata["belongs_to"]) for doc in documents]))
        return {"used_docs": used_docs_names}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("analyze_query_for_category", self.analyze_query_for_category)
        workflow.add_node("factual_query_strategy", self.factual_query_strategy)
        workflow.add_node("analytical_query_strategy", self.analytical_query_strategy)
        workflow.add_node("opinion_query_strategy", self.opinion_query_strategy)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("get_neighboring_docs", self.get_neighboring_docs)
        workflow.add_node("answer_without_context", self.answer_without_context)
        workflow.add_node("generate_answer_with_retrieve_context", self.generate_answer_with_retrieve_context)
        workflow.add_node("check_answer_for_correctness", self.check_answer_for_correctness)
        workflow.add_node("add_source_docs_names", self.add_source_docs_names)

        workflow.add_edge(START, "analyze_query_for_category")
        workflow.add_conditional_edges(
            "analyze_query_for_category",
            self.choose_query_strategy,
            {
                "Factual": "factual_query_strategy",
                "Analytical": "analytical_query_strategy",
                "Opinion": "opinion_query_strategy",
            }
        )

        workflow.add_edge("factual_query_strategy", "retrieve_documents")
        workflow.add_edge("analytical_query_strategy", "retrieve_documents")
        workflow.add_edge("opinion_query_strategy", "retrieve_documents")

        workflow.add_edge("retrieve_documents", "get_neighboring_docs")

        workflow.add_conditional_edges(
            "get_neighboring_docs",
            self.checking_possibility_responses,
            {
                "да": "generate_answer_with_retrieve_context",
                "нет": "answer_without_context"
            }
        )

        workflow.add_edge("generate_answer_with_retrieve_context", "check_answer_for_correctness")
        workflow.add_edge("check_answer_for_correctness", "add_source_docs_names")
        workflow.add_edge("add_source_docs_names", END)

        workflow.add_edge("answer_without_context", END)

        return workflow.compile()

    def __call__(self, *args, **kwargs):
        return self.app


if __name__ == "__main__":
    from src.telegram_bot.services.RetrieverService import CustomRetriever, embeddings
    from src.telegram_bot.langchain_model_init import model
    from langchain_chroma import Chroma
    from pprint import pprint

    vec_store = Chroma(
        collection_name="example_pro",
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
    retriever.vectorstore.add_documents(docs)

    agent = RagAgent(model, retriever)

    while True:
        input_question = input("Введите сообщение: ")

        if input_question != "q":
            inputs = {"question": input_question}
            result = agent().invoke(inputs)
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
