import os

from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
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


def get_pdf_content(path: str) -> str:
    pdf_reader = PdfReader(path)

    raw_text = ""
    for _, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    return raw_text


def read_pdf_folder(folder_path: str) -> List[str]:
    pdfs: List[str] = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdfs.append(get_pdf_content(os.path.join(folder_path, file)))
    return pdfs


def split_pdfs(folder_path: str) -> List[str]:
    splitted_documents: List[str] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pdfs = read_pdf_folder(folder_path)

    for pdf_content in pdfs:
        splitted_documents.extend(text_splitter.split_text(pdf_content))

    return splitted_documents


def embed_pdfs(folder_path: str):
    splitted_pdfs = split_pdfs(folder_path)

    vector_db = SupabaseVectorStore.from_texts(
        client=supabase_client,
        table_name=table_name,
        query_name=query_name,
        texts=splitted_pdfs,
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
