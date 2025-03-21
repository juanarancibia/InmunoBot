from pydantic import BaseModel, Field


class HallucinationDetector(BaseModel):
    is_hallucination: bool = Field(
        default=False, description="Whether the response is a hallucination"
    )


DEFAULT_ANSWER = (
    "No tengo la respuesta para eso! \n\n"
    "Puedo responderte solamente sobre la bibliografía de Inmunología"
)
