from typing import Annotated, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class InputState(TypedDict):
    """Input state for the RAG model."""

    messages: Annotated[list[AnyMessage], add_messages]


class OutputState(InputState):
    """Output state for the RAG model."""

    pass


class OverallState(InputState, OutputState):
    """Overall state for the RAG model."""

    context: str
