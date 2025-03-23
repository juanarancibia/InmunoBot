from pydantic import BaseModel, Field


class HallucinationDetector(BaseModel):
    is_hallucination: bool = Field(
        default=False, description="Whether the response is a hallucination"
    )


class QueryGenerator(BaseModel):
    """Generate search queries for document retrieval"""

    queries: list[str] = Field(
        default_factory=list,
        min_length=2,
        max_length=5,
        description="3 to 5 search queries to retrieve documents based on user message",
    )


DEFAULT_ANSWER = (
    "No tengo la respuesta para eso! \n\n"
    "Puedo responderte solamente sobre la bibliografía de Inmunología"
)
