import os

from langchain_core.documents import Document


class DocumentsGetterService:
    @staticmethod
    def get_source_document(collection_name: str, doc_id: str, belongs_to: str, doc_number: str) -> Document:
        with open(
                rf"/home/alex/PycharmProjects/pythonProject/src/users_directory/{collection_name}/{belongs_to}/{doc_id}_{doc_number}.txt",
                'r') as f:
            content = f.readlines()
            doc = Document(page_content="".join(content))
        return doc

    @staticmethod
    def get_document_by_user_id_section_and_number(user_id: str, section: str, doc_number: str) -> Document:
        files = os.listdir(f"/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/{section}")
        for name in files:
            if name.split("_")[1].replace(".txt", "") == doc_number:
                with open(rf"/home/alex/PycharmProjects/pythonProject/src/users_directory/{user_id}/{section}/{name}",
                          'r') as f:
                    content = f.readlines()
                    doc = Document(page_content="".join(content))
                    return doc
        return Document(page_content="")
