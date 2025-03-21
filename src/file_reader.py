import time
from pathlib import Path
from typing import List
import re
from docling.datamodel.document import ConversionResult
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractOcrOptions
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pprint import pprint


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

    def get_content(self) -> ConversionResult | Exception:
        try:
            result_content = self.document_convertor.convert(self.file_path)
            return result_content
        except Exception as e:
            return e

    def get_markdown(self) -> str:
        result_read = self.get_content()
        result_markdown = result_read.document.export_to_markdown()
        return result_markdown

    @staticmethod
    def get_cleaned_content(content: str) -> str:
        cleaned_content = re.sub(r'\s+|<!-- image -->', ' ', content)
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content)
        return cleaned_content


if __name__ == '__main__':
    t1 = time.time()
    file_reader = FileReader(input_format='pdf',
                             tessdata_path="/usr/share/tesseract-ocr/5/tessdata/",
                             file_path=Path(
                                 "/src/temp_downloads/BQACAgIAAxkBAAOEZ9F9pz1w1EsKaqXc_Zgbiwzv3rsAAr1qAAKk-JBKakiOvKfo8yk2BA.pdf"),
                             language=["eng"])
    result_read = file_reader.get_content()
    result_markdown = result_read.document.export_to_markdown()
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=400, chunk_overlap=0
    )
    split_docs = text_splitter.split_text(result_markdown)
    pprint(split_docs)
