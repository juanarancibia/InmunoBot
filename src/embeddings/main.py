import os
import re

from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import Client, create_client

from lib.llm import AkashModels, get_akash_embedding_model

load_dotenv()

table_name = "documents"
query_name = "match_documents"
embedding_function = get_akash_embedding_model(AkashModels.BAAI_BGE_LARGE)

SUPABASE_API_URL = os.getenv("SUPABASE_API_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

if not SUPABASE_API_URL or not SUPABASE_API_KEY:
    raise ValueError("Supabase URL and API Key must be provided in the .env file")

supabase_client: Client = create_client(SUPABASE_API_URL, SUPABASE_API_KEY)


def load_vector_db():
    return SupabaseVectorStore(
        client=supabase_client,
        table_name=table_name,
        query_name=query_name,
        embedding=embedding_function,
    )


def get_knowledge_db():
    return load_vector_db()


def get_passages(query: str):
    # Ensure query is a string and clean it
    if not isinstance(query, str):
        query = str(query)

    # Clean the query string thoroughly
    query = query.strip()

    # Remove any non-printable or problematic characters that might cause tokenization issues
    query = re.sub(r"[^\x20-\x7E]", "", query)

    # Ensure it's not empty after cleaning
    if not query:
        return []

    try:
        knowledge_db = get_knowledge_db()

        # Use the similarity search
        results = knowledge_db.similarity_search(query, k=5)

        return results

    except Exception as e:
        print(f"Error in get_passages: {e}")
        print(f"Error type: {type(e).__name__}")

        # For now, return empty list to prevent crashes
        # The application should handle this gracefully
        return []
