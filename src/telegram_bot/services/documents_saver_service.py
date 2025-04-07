import os
import shutil
from typing import List
from langchain.schema import Document


class DocumentsSaver:
    @staticmethod
    def __create_user_directory(user_id: str) -> None:
        os.makedirs(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}', exist_ok=True)

    @staticmethod
    def __create_document_section(user_id: str, document_section: str) -> None:
        os.makedirs(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/{document_section}',
                    exist_ok=True)

    @staticmethod
    def __create_file_with_documents_names(user_id: str):
        if os.path.exists(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/user_{user_id}_files'):
            pass
        else:
            with open(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/user_{user_id}_files', 'w') :
                pass
    @staticmethod
    def save_source_docs_ids_names_in_files(user_id: str, docs_id: List[str], documents: List[Document]) -> None:
        """Сохраняет документы в папку пользовател под их id"""
        DocumentsSaver.__create_user_directory(user_id)
        document_section = documents[0].metadata["belongs_to"]
        for doc_id, document in zip(docs_id, documents):
            document_position = document.metadata["doc_number"]
            DocumentsSaver.__create_document_section(user_id, document_section)
            path = f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/{document_section}/{doc_id}_{document_position}.txt'
            with open(path, 'w') as file:
                file.write(document.page_content)

    @staticmethod
    def clear_user_directory(user_id: str) -> None:
        """Удаляет папку с фрагментами документов"""
        DocumentsSaver.__create_user_directory(user_id)
        shutil.rmtree(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}')

    @staticmethod
    def check_exist_user_directory(user_id: str) -> bool:
        """Проверяет сущуствование папки пользователя с фрагментами документов"""
        if os.path.exists(f"/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}"):
            return True
        return False

    @staticmethod
    def add_file_id_with_name_in_file(user_id: str, file_id: str, file_name: str) -> None:
        """Добавляет новый файл с его id и названием"""
        DocumentsSaver.__create_file_with_documents_names(user_id)
        path = f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/user_{user_id}_files'
        with open(path, "a") as file:
            file.write(f"{file_id}::{file_name}")
            file.write("\n")

    @staticmethod
    def delete_file_with_files_ids_names(user_id: str) -> None:
        DocumentsSaver.__create_file_with_documents_names(user_id)
        os.remove(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/user_{user_id}_files')
