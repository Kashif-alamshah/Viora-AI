from typing import TypedDict, Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class Task_State(TypedDict):
    instruction: str
    query: str
    task: str
    image_type: str
    output: str
    summary: str  # <-- NEW: Field to hold the running conversation summary
    messages: Annotated[list, add_messages]
    whole_messages: Annotated[list[BaseMessage], add_messages]

class router_state(BaseModel):
    instruction: str
    query: str