import os
import shutil
from typing import List
from langchain.schema import Document


class DocumentsSaver:
    @staticmethod
    def __create_user_directory(user_id: str) -> None:
        os.makedirs(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}', exist_ok=True)

    def save_source_docs_in_files(self, user_id: str, docs_id: List[str], documents: List[Document]) -> None:
        """Сохраняет документы в папку пользовател под их id"""
        self.__create_user_directory(user_id)
        for doc_id, document in zip(docs_id, documents):
            document_section = document.metadata["belongs_to"]
            document_position = document.metadata["doc_number"]
            path = f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}/{doc_id}/{document_section}/{document_position}.txt'
            with open(path, 'w') as file:
                file.write(document)

    def clear_user_directory(self, user_id: str) -> None:
        """Удаляет папку с фрагментами документов"""
        self.__create_user_directory(user_id)
        shutil.rmtree(f'/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}')

    @staticmethod
    def check_exist_user_directory(user_id: str) -> bool:
        """Проверяет сущуствование папки пользователя с фрагментами документов"""
        if os.path.exists(f"/home/alex/PycharmProjects/pythonProject/src/users_directory/user_{user_id}"):
            return True
        return False
