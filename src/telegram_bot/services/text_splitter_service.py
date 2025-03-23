from typing import List
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitterService:
    def __init__(self,
                 model_id: str = "microsoft/Phi-3-mini-4k-instruct",
                 chunk_size: int = 400,
                 chunk_overlap: int = 100) -> None:
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Создает TextSplitter с заданными параметрами(размер фрагмента и перекрытие)"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        text_splitter = (RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap))
        return text_splitter

    @staticmethod
    def get_split_documents(content: str) -> List[str]:
        """Разделяет текст на фрагменты, возвращет список фрагментов"""
        if len(content) <= 1500:
            return [content]
        elif 1500 < len(content) <= 6000:
            text_splitter = TextSplitterService(chunk_size=500, chunk_overlap=100).create_text_splitter()
            cleaned_split_docs = text_splitter.split_text(content)
            return cleaned_split_docs
        else:
            text_splitter = TextSplitterService(chunk_size=700, chunk_overlap=150).create_text_splitter()
            cleaned_split_docs = text_splitter.split_text(content)
            return cleaned_split_docs
