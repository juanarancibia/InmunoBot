from typing import Dict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import BaseMessage

from embeddings.main import get_passages
from src.lib.llm import (
    AkashModels,
    get_akash_chat_model,
    get_structured_output_with_retry,
    remove_think_tokens,
)
from src.rag.models import DEFAULT_ANSWER, HallucinationDetector
from src.rag.prompt import (
    HALLUCINATION_DETECTOR_PROMPT,
    RESPONSE_GENERATION_PROMPT,
    TRANSLATE_QUESTION_PROMPT,
)
from src.rag.state import InputState, OutputState, OverallState


def retrieve_passages(state: OverallState):
    question = state.get("messages", "")[-1].content
    reasoner_model = get_akash_chat_model(AkashModels.DEEPSEEK_R1_14B, 0.6)

    translated_question = remove_think_tokens(
        reasoner_model.invoke(
            TRANSLATE_QUESTION_PROMPT.format(question=question)
        ).content
    )

    passages = get_passages(translated_question)

    state["context"] = passages

    return state


def generate_response(state: OverallState) -> Dict[str, str]:
    reasoner_model = get_akash_chat_model(AkashModels.DEEPSEEK_R1_32B, 0.6)
    user_message = state.get("messages", "")[-1]
    prompt = RESPONSE_GENERATION_PROMPT.format(
        context=state.get("context", ""),
        previous_messages=state.get("messages", [""])[:-1],
        question=user_message.content
        if hasattr(user_message, "content")
        else user_message,
    )

    response = reasoner_model.invoke(prompt)
    response.content = remove_think_tokens(response.content)

    return {"response": response}


def hallucination_detector(state: OverallState):
    hallucination_detector = get_structured_output_with_retry(
        HallucinationDetector,
        HALLUCINATION_DETECTOR_PROMPT.format(
            response=state.get("response", "").content,
            documents=state.get("context", ""),
        ),
    )

    if hallucination_detector.is_hallucination:
        return {"messages": [BaseMessage(content=DEFAULT_ANSWER)]}

    return {"messages": state.get("response", "")}


def get_workflow():
    graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)

    graph_builder.add_node("retrieve_passages", retrieve_passages)  # type: ignore
    graph_builder.add_node("generate_response", generate_response)  # type: ignore
    graph_builder.add_node("hallucination_detector", hallucination_detector)  # type: ignore

    graph_builder.add_edge(START, "retrieve_passages")
    graph_builder.add_edge("retrieve_passages", "generate_response")
    graph_builder.add_edge("generate_response", "hallucination_detector")
    graph_builder.add_edge("hallucination_detector", END)

    return graph_builder.compile()  # type: ignore


def invoke_graph(messages, callables):
    runnable = get_workflow()

    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")

    # Invoke the graph with the current messages and callback configuration
    return runnable.invoke({"messages": messages}, config={"callbacks": callables})
