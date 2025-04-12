import tesserocr
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractOcrOptions
from docling.datamodel.settings import settings
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

import os

from typing import NamedTuple


class AnswerAndSourceDocs(NamedTuple):
    answer: str
    source: str


IMAGE_RESOLUTION_SCALE = 2.0
os.environ['HF_TOKEN'] = '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'
print(tesserocr.get_languages("/usr/share/tesseract-ocr/5/tessdata"))

pipeline_options = PdfPipelineOptions(

    do_table_structure=False,  # Enable table structure detection
    do_ocr=True,  # Enable OCR
    # full page ocr and language selection
    # ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=["en"]),  # Use EasyOCR for OCR
    ocr_options=TesseractOcrOptions(path="/usr/share/tesseract-ocr/5/tessdata/", force_full_page_ocr=True, lang=["rus"]),
    # Uncomment to use Tesseract for OCR
    # ocr_options = OcrMacOptions(force_full_page_ocr=True, lang=['en-US']),
    table_structure_options=dict(
        do_cell_matching=False,  # Use text cells predicted from table structure model
        mode=TableFormerMode.ACCURATE  # Use more accurate TableFormer model
    ),
)

doc_converter_global = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

settings.debug.profile_pipeline_timings = True
doc_filename = Path("/home/alex/PycharmProjects/pythonProject/content/Andrey.pdf")

# Convert the document

# result = doc_converter_global.convert("page_25.pdf")
result = doc_converter_global.convert(Path(doc_filename))

result = result.document.export_to_markdown()
print(result)