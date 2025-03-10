import time
from pathlib import Path
from typing import List, NamedTuple
from docling.datamodel.document import ConversionResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractOcrOptions

from src.langchain_model_init import model
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from src.LangChainVec import get_retriever
from pprint import pprint
from huggingface_hub import hf_hub_download


class SummarizeContentAndDocs(NamedTuple):
    summary: List[str]
    source_docs: List[str]


def exponential_backoff(retries, initial_delay=1):
    """Функция для расчета экспоненциальной задержки."""
    return min(initial_delay * (2 ** retries), 60)  # Максимальная задержка — 60 секунд


class FileReader:
    def __init__(self,
                 input_format: str,
                 file_path: Path,
                 tessdata_path: str,
                 language: List[str],
                 do_table_structure: bool = True,
                 generate_page_images: bool = True,
                 generate_picture_images: bool = True,
                 images_scale: float = 2.0) -> None:

        self.file_path = file_path
        self.input_format = input_format
        self.tessdata_path = tessdata_path
        self.language = language
        self.do_table_structure = do_table_structure
        self.generate_page_images = generate_page_images
        self.generate_picture_images = generate_picture_images
        self.images_scale = images_scale
        self.document_convertor = self._init_document_converter()

    def _init_document_converter(self) -> DocumentConverter | Exception:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=self.do_table_structure,
            do_ocr=True,
            ocr_options=TesseractOcrOptions(path=self.tessdata_path, force_full_page_ocr=True,
                                            lang=self.language),
            table_structure_options=dict(
                do_cell_matching=False,
                mode=TableFormerMode.ACCURATE
            ),
            generate_page_images=self.generate_page_images,
            generate_picture_images=self.generate_picture_images,
            images_scale=self.images_scale,
        )

        match self.input_format:
            case 'pdf':
                return DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
            # case 'docx':
            #     return DocumentConverter(
            #         format_options={InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options)})
            # case 'pptx':
            #     return DocumentConverter(
            #         format_options={InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pipeline_options)})
            # case 'html':
            #     return DocumentConverter(
            #         format_options={InputFormat.HTML: HTMLFormatOption(pipeline_options=pipeline_options)})
            # case 'image':
            #     return DocumentConverter(
            #         format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)})
            # case 'asciidoc':
            #     return DocumentConverter(
            #         format_options={InputFormat.ASCIIDOC: AsciiDocFormatOption(pipeline_options=pipeline_options)})
            # case _:
            #     return Exception(f'Invalid format: {self.input_format}')

    def get_content(self) -> ConversionResult | Exception:
        try:
            result_content = self.document_convertor.convert(self.file_path)
            return result_content
        except Exception as e:
            return e

    @staticmethod
    def get_summarize_docs_content(split_docs: List[str],
                                   model: BaseChatModel) -> SummarizeContentAndDocs | Exception:
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
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        batch_size = int(len(split_docs) * 0.2)
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


if __name__ == '__main__':
    t1 = time.time()
    # ИЗВЛЕЧЕНИЕ ТЕКСТА
    file_reader = FileReader(input_format='pdf',
                             tessdata_path="/usr/share/tesseract-ocr/5/tessdata/",
                             file_path=Path("/home/alex/PycharmProjects/pythonProject/content/gold_fish.pdf"),
                             language=["rus"])
    result_read = file_reader.get_content()
    result_markdown = result_read.document.export_to_markdown()
    # print("----------RESULT MARKDOWN----------")
    # pprint(result_markdown)

    # РАЗДЕЛЕНИЕ ТЕКСТА НА ЧАСТИ
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=400, chunk_overlap=0
    )
    split_docs = text_splitter.split_text(result_markdown)
    # print("----------SPLIT DOCS----------")
    # print('count docs', len(split_docs))
    pprint(split_docs)

    # СЖАТИЕ ФРАГМЕНТОВ ТЕКСТА

    result_summary = file_reader.get_summarize_docs_content(split_docs, model)
    # print("----------RESULT SUMMARY----------")
    # print(result_summary)
    # print('count sum docs', len(result_summary.summary))
    pprint(result_summary.summary)

    # ДОБАВЛЕНИЕ ФРАГМЕНТОВ В ВЕКТОРНУЮ БАЗУ
    doc_ids = [str(i) for i in range(len(split_docs))]
    summarize_docs = [
        Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) for i, summary in
        enumerate(result_summary.summary)
    ]
    retriever = get_retriever()
    retriever.vectorstore.add_documents(summarize_docs)
    retriever.docstore.mset(list(zip(doc_ids, split_docs)))
    print("----------RESULT RETRIEVER----------")
    res = retriever.invoke("Что попросил сделать старик старуху?")
    pprint(res)
    t2 = time.time()
    print(t2 - t1)
