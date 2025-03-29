import base64
import io
import os
from typing import Any

import fitz
import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from PIL import Image
from supabase import Client, create_client
from transformers import AutoModelForVision2Seq, AutoProcessor

from src.lib.llm import AkashModels, get_akash_embedding_model

load_dotenv()

table_name = "documents"
query_name = "match_documents"
embedding_function = get_akash_embedding_model(AkashModels.BAAI_BGE_LARGE)

SUPABASE_API_URL = os.getenv("SUPABASE_API_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

if not SUPABASE_API_URL or not SUPABASE_API_KEY:
    raise ValueError("Supabase URL and API Key must be provided in the .env file")

supabase_client: Client = create_client(SUPABASE_API_URL, SUPABASE_API_KEY)

# Initialize the model
# Refer to: https://huggingface.co/ds4sd/SmolDocling-256M-preview
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")

model = AutoModelForVision2Seq.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    torch_dtype=torch.bfloat16,
).to(DEVICE)


def pdf_page_to_base64(pdf_document: Any, page_number: int):
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue())


def get_pdf_as_image_list(pdf_path: str) -> list[bytes]:
    pdf_document = fitz.open(pdf_path)

    return [
        pdf_page_to_base64(pdf_document, page_number)
        for page_number in range(pdf_document.page_count)
    ]


def get_docling_md_from_imgs(images_list: list[bytes]) -> DoclingDocument:
    # Prepare inputs
    images = [
        Image.open(io.BytesIO(base64.b64decode(image_bytes)))
        for image_bytes in images_list
    ]

    all_doctags = []

    # Process each image individually
    for img in images:
        # Create input messages for each image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."},
                ],
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        inputs = inputs.to(DEVICE)

        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]

        doctag = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
        )[0].lstrip()

        all_doctags.append(doctag)

    # Create a docling document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(all_doctags, images)
    doc = DoclingDocument(name="Document")

    doc.load_from_doctags(doctags_doc)

    return doc.export_to_markdown()


def split_markdown(md: str) -> list[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(md)

    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)

    return splits


def embed_pdfs(documents: list[Document]):
    vector_db = SupabaseVectorStore.from_documents(
        client=supabase_client,
        table_name=table_name,
        query_name=query_name,
        documents=documents,
        embedding=embedding_function,
    )

    return vector_db


if __name__ == "__main__":
    pdfs_files_paths = os.listdir("./pdfs")

    documents = []

    for pdf_file_path in pdfs_files_paths:
        pdf_path = os.path.join("./pdfs", pdf_file_path)
        markdown_text = get_docling_md_from_imgs(get_pdf_as_image_list(pdf_path))
        splitted_documents = split_markdown(markdown_text)
        documents.extend(splitted_documents)
        print(f"Loaded {len(splitted_documents)} documents from {pdf_file_path}")

    # Embed documents
    embed_pdfs(documents)
    print(f"Loaded {len(documents)} documents in total")
