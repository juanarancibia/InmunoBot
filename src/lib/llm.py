import os
from enum import Enum

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class AkashModels(Enum):
    DEEPSEEK_R1_32B = "DeepSeek-R1-Distill-Qwen-32B"
    DEEPSEEK_R1_14B = "DeepSeek-R1-Distill-Qwen-14B"
    LLAMA_4 = "Meta-Llama-4-Maverick-17B-128E-Instruct-FP8"
    BAAI_BGE_LARGE = "BAAI-bge-large-en-v1-5"


def get_akash_chat_model(model: AkashModels, temperature: float):
    """
    Returns the Akash Chat instance
    """
    return ChatOpenAI(
        model=model.value,
        temperature=temperature,
        base_url="https://chatapi.akash.network/api/v1",
        api_key=os.environ.get("AKASH_API_KEY", ""),
    )


def get_structured_output_with_retry(
    structured_schema: type, value: str
) -> type | None:
    """
    Returns the structured output response with retry, or None if validation fails
    """
    result = None
    models_to_try = [
        AkashModels.DEEPSEEK_R1_14B,
        AkashModels.DEEPSEEK_R1_32B,
    ]

    for model in models_to_try:
        try:
            print(f"Getting structured output from Akash model {model.name}")

            akash_chat = get_akash_chat_model(model, 0)
            structured_akash = akash_chat.with_structured_output(structured_schema)
            result = structured_akash.invoke(value)

            return result
        except Exception as e:
            print(f"Unexpected error with Akash model {model.name}: {e}")

    return result


def get_akash_embedding_model(model: AkashModels):
    """
    Returns the Akash Embedding instance
    """
    return OpenAIEmbeddings(
        model=model.value,
        base_url="https://chatapi.akash.network/api/v1",
        api_key=os.environ.get("AKASH_API_KEY", ""),
    )


def remove_think_tokens(result: str):
    if "</think>" in result:
        result = result.split("</think>")[1]
    return result
