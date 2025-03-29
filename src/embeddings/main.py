import os

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
    knowledge_db = get_knowledge_db()

    return knowledge_db.similarity_search(query, k=5)
