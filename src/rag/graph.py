from typing import Dict

from langgraph.graph import END, START, StateGraph

from lib.llm import AkashModels, get_akash_model, remove_think_tokens
from rag.prompt import RESPONSE_GENERATION_PROMPT
from rag.state import InputState, OutputState, OverallState


def retrieve_passages(state: OverallState):
    pass


def generate_response(state: OverallState) -> Dict[str, str]:
    reasoner_model = get_akash_model(AkashModels.AKASH_DEEPSEEK_R1_32B, 0.6)
    user_message = state.get("messages", "")[-1]
    prompt = RESPONSE_GENERATION_PROMPT.format(
        context="",
        previous_messages=state.get("messages", [""])[:-1],
        question=user_message.content
        if hasattr(user_message, "content")
        else user_message,
    )

    response = reasoner_model.invoke(prompt)
    response.content = remove_think_tokens(response.content)

    return {"messages": [response]}


def get_workflow():
    graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)

    graph_builder.add_node("retrieve_passages", retrieve_passages)  # type: ignore
    graph_builder.add_node("generate_response", generate_response)  # type: ignore

    graph_builder.add_edge(START, "retrieve_passages")
    graph_builder.add_edge("retrieve_passages", "generate_response")
    graph_builder.add_edge("generate_response", END)

    return graph_builder.compile()  # type: ignore


def invoke_graph(messages, callables):
    runnable = get_workflow()

    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")

    # Invoke the graph with the current messages and callback configuration
    return runnable.invoke({"messages": messages}, config={"callbacks": callables})
