"""Custom embedding implementation to work around tokenization issues"""

import json
import os
from typing import List

import requests
from langchain_core.embeddings import Embeddings


class CustomAkashEmbeddings(Embeddings):
    """Custom Akash embedding implementation that ensures raw text is sent to the API"""

    def __init__(self, model: str = "BAAI-bge-large-en-v1-5"):
        self.model = model
        self.api_key = os.environ.get("AKASH_API_KEY", "")
        self.base_url = "https://chatapi.akash.network/api/v1"

        if not self.api_key:
            raise ValueError("AKASH_API_KEY environment variable is required")

    def _make_embedding_request(self, texts: List[str]) -> List[List[float]]:
        """Make direct API request to Akash embeddings endpoint"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Ensure all texts are clean strings
        clean_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            # Clean the text of any problematic characters
            clean_text = text.strip()
            clean_texts.append(clean_text)

        payload = {
            "model": self.model,
            "input": clean_texts,
            "encoding_format": "float",
        }

        print(f"Making embedding request for {len(clean_texts)} texts")
        print(f"First text sample: {repr(clean_texts[0][:100])}")

        try:
            response = requests.post(
                f"{self.base_url}/embeddings", headers=headers, json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                print(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings
            else:
                print(f"API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                raise Exception(f"Embedding API request failed: {response.status_code}")

        except requests.RequestException as e:
            print(f"Request error: {e}")
            raise Exception(f"Network error during embedding request: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self._make_embedding_request(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embeddings = self._make_embedding_request([text])
        return embeddings[0] if embeddings else []
