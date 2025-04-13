import time
from typing import NamedTuple, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough


class SummarizeContentAndDocs(NamedTuple):
    summary_texts: List[str]
    source_docs: List[str]


def exponential_backoff(retries, initial_delay=1):
    """Функция для расчета экспоненциальной задержки."""
    return min(initial_delay * (2 ** retries), 60)  # Максимальная задержка — 60 секунд


class LLMModelService:
    def __init__(self, model: BaseChatModel):
        self.model = model

    def _get_answer(self, prompt_text, documents: List[str]) -> SummarizeContentAndDocs:
        """Генерирует ответ модели по заданному prompt, который содержит поле element"""
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | self.model | StrOutputParser()
        batch_size = max(int(len(documents) * 0.2), 1)
        batches_split_docs = [documents[i: i + batch_size] for i in range(0, len(documents), batch_size)]
        retries, max_retries, result_text_sum = 0, 5, []
        for batch in batches_split_docs:
            while retries < max_retries:
                try:
                    text_sum = (summarize_chain
                                .with_retry(wait_exponential_jitter=True, stop_after_attempt=6)
                                .invoke(batch))
                    result_text_sum.append(text_sum)
                    print("TEXT SUM: {}".format(text_sum))
                    retries = 0
                    break
                except Exception as e:
                    print("Exception: {}".format(e))
                    retries += 1
                    delay = exponential_backoff(retries)
                    time.sleep(delay)
        return SummarizeContentAndDocs(result_text_sum, documents)

    def get_summarize_docs(self, documents: List[str]) -> str:
        """Создает краткое описание для документов
        Возвращает общее краткое содержание
        """
        prompt_text = """    
                  Вы помощник, который должен передавать краткое содержание текста, сохраняя все важные детали.\n
                  Предоставьте краткое содержание,чтобы сохранить все важные моменты и детали.\n
                  ВАжные детали  - имена персонажей, названия чего либо, действия, время года/суток\n
                  Сильно не сокращайте текст, оставьте как можно можно больше деталей. удалите только маловажные моменты.\n
                  Для резюмирования используйте ТОЛЬКО ДАННЫЕ из предложенного фрагмента,\n
                  не используй свои знания или данные из других фрагментов, которых нет в предложенном.\n
                  Не начинайте свое сообщение словами «Вот резюме», "В этом фрагменте", 'В этом документе' или чем то дургим\n
                  и не используюте отдельние "Вопросы к фрагменту" и подобное.\n
                  Просто дайте краткое содержание.

                  Исходный текст: {element}

                  """
        answer = self._get_answer(prompt_text, documents)
        return "".join(answer.summary_texts)

    def get_summarize_docs_with_questions(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        """Создает краткое описание к документам и добавлет вопросы к каждому фрагменту
        Возвращает краткие содержания и исходные фрагменты
        """
        prompt_text = """    
           Вы помощник, который должен передавать краткое содержание текста, сохраняя все важные детали.\n
           Предоставьте краткое содержание,чтобы сохранить все важные моменты и детали.\n
           Сильно не сокращайте текст, оставьте как можно можно больше деталей. удалите только маловажные моменты.\n
           Для резюмирования используйте ТОЛЬКО ДАННЫЕ из предложенного фрагмента,\n
           не используй свои знания или данные из других фрагментов, которых нет в предложенном.\n
           Также напишите несколько вопросов, которые пользователь может задать к этому фрагменту.\n

           Отвечайте только краткое содержание и вопросы к нему, без дополнительных комментариев. \n
           Не нужно отделять вопросы от краткого содержания. Они должны идти подряд.\n
           Не начинайте свое сообщение словами «Вот резюме», "В этом фрагменте", 'В этом документе' или чем то дургим\n
           и не используюте отдельние "Вопросы к фрагменту" и подобное.\n
           Просто дайте краткое содержание и вопросы. Вопросы всегда должны быть отделены от текста с помощью слова "Вопросы:"\n

           text chunk: {element}

           """
        answer = self._get_answer(prompt_text, split_docs)
        return answer

    def get_super_brief_content(self, content: str, max_word=70) -> str:
        """Вовзращает краткое содержание размер max_word"""
        prompt_text = """
        Проанализируй предоставленный текст и создай его краткое содержание, сохраняя только ключевые идеи и основную суть. Удали всё, что не относится к содержанию: примеры, повторы, метафоры, эмоциональные оценки, второстепенные детали и отступления. Суммаризация должна быть объективной, чёткой и лаконичной.\n
        **Требования:**\n
        1. Передай главную мысль текста.\n
        2. Если текст содержит несколько важных тезисов, перечисли их кратко (не более 3-5 пунктов).\n
        3. Не включай субъективные мнения автора и примеры.\n
        4. Используй нейтральный стиль, без вводных слов и оценочных суждений.\n        
        Теперь проанализируй следующий текст:\n
        {context}\n        
        НЕ используй более {max_word} слов
        """

        chain = ChatPromptTemplate.from_template(prompt_text) | RunnablePassthrough(
            lambda x: print('prompt', x)) | self.model | StrOutputParser()
        return chain.invoke({"max_word": max_word, "context": content})
