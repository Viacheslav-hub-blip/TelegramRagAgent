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

    do_table_structure=True,  # Enable table structure detection
    do_ocr=True,  # Enable OCR
    # full page ocr and language selection
    # ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=["en"]),  # Use EasyOCR for OCR
    ocr_options=TesseractOcrOptions(path="/usr/share/tesseract-ocr/5/tessdata/", force_full_page_ocr=True, lang=["eng"]),
    # Uncomment to use Tesseract for OCR
    # ocr_options = OcrMacOptions(force_full_page_ocr=True, lang=['en-US']),
    table_structure_options=dict(
        do_cell_matching=False,  # Use text cells predicted from table structure model
        mode=TableFormerMode.ACCURATE  # Use more accurate TableFormer model
    ),
    generate_page_images=True,  # Enable page image generation
    generate_picture_images=True,  # Enable picture image generation
    images_scale=IMAGE_RESOLUTION_SCALE,  # Set image resolution scale (scale=1 corresponds to a standard 72 DPI image)
)

doc_converter_global = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

settings.debug.profile_pipeline_timings = True
doc_filename = Path("/home/alex/PycharmProjects/pythonProject/content/page_25.pdf")

# Convert the document

# result = doc_converter_global.convert("page_25.pdf")
result = doc_converter_global.convert(Path(doc_filename))

result = result.document.export_to_markdown()

model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=800, chunk_overlap=0
)

split_docs = text_splitter.split_text(result)
print('spli texts', split_docs)
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary_with_questions of the table or text.

Respond only with the summary_with_questions, no additionnal comment.
Do not start your message by saying "Here is a summary_with_questions" or anything like that.
Just give the summary_with_questions as it is.

text chunk: {element}

"""
# prompt = ChatPromptTemplate.from_template(prompt_text)
#
# summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
#
# text_sum = summarize_chain.batch(split_docs)
#
# # doc_ids = [str(uuid.uuid4()) for _ in split_docs]
# doc_ids = [str(i) for i in range(len(split_docs))]
#
# summarize_docs = [
#     Document(page_content=summary_with_questions, metadata={"doc_id": doc_ids[i]}) for i, summary_with_questions in enumerate(text_sum)
# ]
# retriever.vectorstore.add_documents(summarize_docs)
# retriever.docstore.mset(list(zip(doc_ids, split_docs)))


# def get_original_source(docs):
#     source_texts = []
#     for doc in docs:
#         id_doc = doc[0].metadata["doc_id"]
#         text = retriever.docstore.mget([id_doc])[0]
#         source_texts.append(text)
#     return {"source_texts": source_texts}
#
#
# def build_prompt(kwargs):
#     context_docs = kwargs["context"]
#     user_question = kwargs["user_question"]
#     context_text = ""
#     if len(context_docs) > 0:
#         for doc in context_docs:
#             context_text += doc[0].page_content
#     prompt_template = f"""
#        Answer the question based only on the following context, which can include text, tables, and the below image.
#        If the context is empty, answer based on your knowledge. For example: I can't answer based on the context, but here's what I know:...
#        Context: {context_text}
#        Question: {user_question}
#        """
#     return ChatPromptTemplate.from_messages(
#         [
#             HumanMessage(prompt_template)
#         ]
#     )
#
#
# @chain
# def custom_chain(user_question) -> AnswerAndSourceDocs:
#     context = retriever.invoke(user_question)
#     similar_documents = [doc for doc in context if doc[1] <= 200]
#     result_prompt = RunnableLambda(build_prompt).invoke({"context": similar_documents, "user_question": user_question})
#
#     if len(similar_documents) > 0:
#         source_docs = "\n".join([doc[0].metadata['source_doc'] for doc in similar_documents][0])
#     else:
#         source_docs = "knowledge based"
#     sub_chain = model | StrOutputParser()
#     return AnswerAndSourceDocs(sub_chain.invoke(result_prompt), source_docs)
#
#
# query = "who is Elon Musk?"
# answer, source = custom_chain.invoke(query)
# print(answer)
# pprint(source)
