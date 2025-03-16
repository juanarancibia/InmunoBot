from typing import TypedDict


class InputState(TypedDict):
    """Input state for the RAG model."""
    messages: list[str]

class OutputState(TypedDict):
    """Output state for the RAG model."""
    response: str

class OverallState(InputState, OutputState):
    """Overall state for the RAG model."""
    context: str