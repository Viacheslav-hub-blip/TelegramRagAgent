import pymupdf
import re


class PDFReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_content(self) -> list[str]:
        document = pymupdf.open(self.file_path)
        content = []
        for page in document:
            text = page.get_text()
            content.append(text)
        return content

    def get_cleaned_content(self) -> str:
        content = self.get_content()
        cleaned_content = []
        for c in content:
            cleaned_text = re.sub(r'^.\s\n', '', c, flags=re.MULTILINE)
            cleaned_text = re.sub(r'^.\n', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
            cleaned_content.append(cleaned_text)
        return "".join(cleaned_content)
