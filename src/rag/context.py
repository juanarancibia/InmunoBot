import concurrent.futures

from langgraph.graph.message import AnyMessage

from embeddings.main import get_passages
from lib.llm import AkashModels, get_akash_chat_model, remove_think_tokens
from rag.models import QueryGenerator
from rag.prompt import TRANSLATE_USER_MESSAGE_PROMPT


def translate_user_message(user_message: AnyMessage):
    reasoner_model = get_akash_chat_model(AkashModels.DEEPSEEK_R1_14B, 0.6)

    translated_message = remove_think_tokens(
        reasoner_model.invoke(
            TRANSLATE_USER_MESSAGE_PROMPT.format(user_message=user_message)
        ).content
    )

    return translated_message


def generate_queries(user_message: str):
    translated_message = translate_user_message(user_message)

    # Divide the message in n queries
    reasoner_model = get_akash_chat_model(AkashModels.DEEPSEEK_R1_14B, 0.6)
    structured_output_model = reasoner_model.with_structured_output(QueryGenerator)

    queries = structured_output_model.invoke(translated_message).queries

    return queries


def reciprocal_rank_fusion(results):
    fused_documents = {}
    k = 60

    for rank, doc in enumerate(results):
        doc_str = str(doc)

        # If the document is not yet in the fused_documents dictionary,
        # add it with an initial score of 0
        if doc_str not in fused_documents:
            fused_documents[doc_str] = 0

        # Update the score of the document using the RRF formula: 1 / (rank + k)
        fused_documents[doc_str] += 1 / (rank + k)

    # final reranked result
    reranked_results = [
        (doc)
        for doc, score in sorted(
            fused_documents.items(), key=lambda x: x[1], reverse=True
        )
    ]

    return reranked_results


def retrieve_and_rerank(queries: list[str]):
    # Use ThreadPoolExecutor to run queries in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Get passages for each query in parallel
        query_results = list(executor.map(get_passages, queries))

        # Flatten the list of lists into a single list
        results = []
        for query_result in query_results:
            results.extend(query_result)

    return reciprocal_rank_fusion(results)


def retrieve_context(user_message: str):
    # Divide the message in n queries
    queries = generate_queries(user_message)

    # Retrieve passages for all the queries
    passages = retrieve_and_rerank(queries)

    return passages[:5]
