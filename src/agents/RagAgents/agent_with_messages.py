from typing import TypedDict, Annotated
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    question: str


class AgentWithHistory:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.state = GraphState

    def call_model(self, state: GraphState):
        summary = state.get("summary", "")
        if summary:
            system_message = f"Краткое содержание истории диалога: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]
