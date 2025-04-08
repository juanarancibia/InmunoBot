from typing import Dict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import BaseMessage

from lib.llm import (
    AkashModels,
    get_akash_chat_model,
    get_structured_output_with_retry,
    remove_think_tokens,
)
from rag.context import retrieve_context
from rag.models import DEFAULT_ANSWER, HallucinationDetector
from rag.prompt import HALLUCINATION_DETECTOR_PROMPT, RESPONSE_GENERATION_PROMPT
from rag.state import InputState, OutputState, OverallState


def retrieve_passages(state: OverallState):
    user_message = state.get("messages", "")[-1].content

    passages = retrieve_context(user_message)

    return {"context": passages}


def generate_response(state: OverallState) -> Dict[str, str]:
    chat_model = get_akash_chat_model(AkashModels.LLAMA_4, 0.5)

    messages = state.get("messages", [""])
    user_message = messages[-1]
    previous_messages = messages[-5:-1] if len(messages) >= 5 else messages[:-1]
    prompt = RESPONSE_GENERATION_PROMPT.format(
        context=state.get("context", ""),
        previous_messages=previous_messages,
        question=user_message.content
        if hasattr(user_message, "content")
        else user_message,
    )

    response = chat_model.invoke(prompt)

    return {"response": response.content}


def hallucination_detector(state: OverallState):
    hallucination_detector = get_structured_output_with_retry(
        HallucinationDetector,
        HALLUCINATION_DETECTOR_PROMPT.format(
            response=state.get("response", "").content,
            documents=state.get("context", ""),
        ),
    )

    if hallucination_detector.is_hallucination:
        response_message = BaseMessage(content=str(DEFAULT_ANSWER), type="ai")
        return {"messages": [response_message]}

    return {"messages": state.get("response", "")}


def get_workflow():
    graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)

    graph_builder.add_node("retrieve_passages", retrieve_passages)
    graph_builder.add_node("generate_response", generate_response)
    graph_builder.add_node("hallucination_detector", hallucination_detector)

    graph_builder.add_edge(START, "retrieve_passages")
    graph_builder.add_edge("retrieve_passages", "generate_response")
    graph_builder.add_edge("generate_response", "hallucination_detector")
    graph_builder.add_edge("hallucination_detector", END)

    return graph_builder.compile()


def invoke_graph(messages, callables):
    runnable = get_workflow()

    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")

    # Invoke the graph with the current messages and callback configuration
    return runnable.invoke({"messages": messages}, config={"callbacks": callables})
