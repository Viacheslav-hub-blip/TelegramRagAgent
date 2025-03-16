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
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        text_splitter = (RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap))
        return text_splitter
