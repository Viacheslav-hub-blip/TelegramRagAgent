import os
import shutil
from typing import List


class DocumentsSaver:
    @staticmethod
    def __check_exist_user_directory(user_id: str) -> None:
        os.makedirs(f'users_directory/{user_id}', exist_ok=True)

    def save_source_docs_in_files(self, user_id: str, docs_id: List[str], documents: List[str]) -> None:
        self.__check_exist_user_directory(user_id)
        for doc_id, document in zip(docs_id, documents):
            path = f'users_directory/{user_id}/{doc_id}.txt'
            with open(path, 'w') as file:
                file.write(document)

    def clear_user_directory(self, user_id: str) -> None:
        self.__check_exist_user_directory(user_id)
        shutil.rmtree(f'users_directory/{user_id}')
