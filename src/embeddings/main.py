import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from typing_extensions import List

from ..lib.llm import AkashModels, get_akash_embedding_model

persist_directory = "knowledge_db"
embedding_function = get_akash_embedding_model(AkashModels.BAAI_BGE_LARGE)


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

    vector_db = Chroma.from_texts(  # type: ignore
        texts=splitted_pdfs,
        embedding=embedding_function,
        persist_directory=persist_directory,
    )

    return vector_db


def load_vector_db():
    return Chroma(
        persist_directory=persist_directory, embedding_function=embedding_function
    )


def get_knowledge_db():
    if not os.path.exists(persist_directory):
        embed_pdfs("pdfs")

    return load_vector_db()


def get_passages(query: str):
    knowledge_db = get_knowledge_db()

    return knowledge_db.similarity_search(query, k=5)
