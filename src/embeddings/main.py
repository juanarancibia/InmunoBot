import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client
from typing_extensions import List

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


def read_pdf_folder(folder_path: str) -> List[Document]:
    loader = PyPDFDirectoryLoader(folder_path)

    docs = loader.load()

    return docs


def split_pdfs(folder_path: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pdfs = read_pdf_folder(folder_path)

    splitted_documents = text_splitter.split_documents(pdfs)

    return splitted_documents


def embed_pdfs(folder_path: str):
    splitted_pdfs = split_pdfs(folder_path)

    vector_db = SupabaseVectorStore.from_documents(
        client=supabase_client,
        table_name=table_name,
        query_name=query_name,
        documents=splitted_pdfs,
        embedding=embedding_function,
    )

    return vector_db


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


if __name__ == "__main__":
    embed_pdfs("pdfs")
    print("PDFs embedded successfully! 🚀")
