from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
import os
from langchain_gigachat.chat_models import GigaChat

os.environ[
    "GIGACHAT_API_PERS"] = "ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmViOTBhNDZmLTAxNzktNDY4Yi04ODljLTc3ZDZhOTA0YmJjZg=="

model = GigaChat(verify_ssl_certs=False,
                 credentials="ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmViOTBhNDZmLTAxNzktNDY4Yi04ODljLTc3ZDZhOTA0YmJjZg==",
                 model="GigaChat-2")
