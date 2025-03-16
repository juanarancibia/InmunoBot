from rag.state import OverallState
from lib.llm import get_akash_model, AkashModels
from rag.prompt import RESPONSE_GENERATION_PROMPT
from langgraph.graph import StateGraph, START, END



def retrieve_passages(state: OverallState):
    pass

def generate_response(state: OverallState):
    reasoner_model = get_akash_model(AkashModels.AKASH_DEEPSEEK_R1_32B, 0.6)
    user_message = state.get("messages", "")[-1]
    prompt = RESPONSE_GENERATION_PROMPT.format(
        context="",
        question=state.get(user_message, ""),
    )

    response = reasoner_model(prompt)

    return { "response": response }

def get_workflow():
    graph_builder = StateGraph()

    graph_builder.add_node("retrieve_passages", retrieve_passages)
    graph_builder.add_node("generate_response", generate_response)

    graph_builder.add_edge(START, "retrieve_passages")
    graph_builder.add_edge("retrieve_passages", "generate_response")
    graph_builder.add_edge("generate_response", END)

    return graph_builder.compile()
    