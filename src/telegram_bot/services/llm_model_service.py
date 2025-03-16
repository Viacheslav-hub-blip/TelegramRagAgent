import time
from typing import NamedTuple, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


class SummarizeContentAndDocs(NamedTuple):
    summary: List[str]
    source_docs: List[str]


def exponential_backoff(retries, initial_delay=1):
    """Функция для расчета экспоненциальной задержки."""
    return min(initial_delay * (2 ** retries), 60)  # Максимальная задержка — 60 секунд


class LLMModelService:
    def __init__(self, model: BaseChatModel):
        self.model = model

    def get_summarize_docs_content(self, split_docs: List[str]) -> SummarizeContentAndDocs:
        prompt_text = """    
           Вы помощник, который должен резюмировать текст.
           Предоставьте резюме,чтобы сохранить все важные моменты и детали.
           Для резюмирования используйте ТОЛЬКО ДАННЫЕ из предложенного фрагмента,
           не используй свои знания или данные из других фрагментов, которых нет в предложенном.
           Также напишите несколько вопросов, которые пользователь может задать к этому фрагменту.

           Отвечайте только резюме и вопросы к нему, без дополнительных комментариев. 
           Не нужно отделять вопросы от краткого содержания. Они должны идти подряд.
           Не начинайте свое сообщение словами «Вот резюме», "В этом фрагменте", 'В этом документе' или чем то дургим
           и не используюте отдельние "Вопросы к фрагменту" и подобное.
           Просто дайте резюме и вопросы.

           text chunk: {element}

           """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | self.model | StrOutputParser()
        batch_size = max(int(len(split_docs) * 0.2), 1)
        batches_split_docs = [split_docs[i: i + batch_size] for i in range(0, len(split_docs), batch_size)]
        retries = 0
        max_retries = 5
        result_text_sum = []
        for batch in batches_split_docs:
            while retries < max_retries:
                try:
                    text_sum = (summarize_chain
                                .with_retry(wait_exponential_jitter=True, stop_after_attempt=3)
                                .invoke(batch))
                    result_text_sum.append(text_sum)
                    retries = 0
                    break
                except Exception as e:
                    print(f"Ошибка: {e}. Попытка {retries + 1} из {max_retries}")
                    retries += 1
                    delay = exponential_backoff(retries)
                    time.sleep(delay)
            else:
                print(f"Не удалось обработать батч: {batch}")
        return SummarizeContentAndDocs(result_text_sum, split_docs)
