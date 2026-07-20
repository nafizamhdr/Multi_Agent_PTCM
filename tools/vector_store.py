"""
vector_store.py
Wrapper ChromaDB untuk keperluan RAG (dokumen SOP) dan semantic case search
(histori maintenance & komplain CS).

Embedding default: TF-IDF (scikit-learn) — ringan, jalan 100% offline, cukup untuk
skeleton/demo. Untuk kualitas retrieval yang lebih baik saat production, ganti
`TfidfEmbeddingFunction` dengan sentence-transformers (lihat komentar di bawah).
"""

import os
import joblib
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TfidfEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function custom untuk ChromaDB berbasis TF-IDF.
    Di-fit sekali saat koleksi dibuat, lalu dipakai untuk transform query baru.

    NOTE (untuk upgrade production):
    Ganti class ini dengan wrapper sentence-transformers, contoh:

        from sentence_transformers import SentenceTransformer
        class STEmbeddingFunction(EmbeddingFunction):
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)
            def __call__(self, input: Documents) -> Embeddings:
                return self.model.encode(input).tolist()

    Embedding transformer-based akan menangkap makna semantik jauh lebih baik
    daripada TF-IDF yang hanya berbasis kemunculan kata.
    """

    def __init__(self, max_features: int = 512, vectorizer_path: str = None):
        self.vectorizer_path = vectorizer_path
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

        # Coba load vectorizer yang sudah pernah di-fit sebelumnya (persisten antar proses)
        if vectorizer_path and os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
            self._fitted = True

    def fit(self, corpus: list[str]):
        self.vectorizer.fit(corpus)
        self._fitted = True
        if self.vectorizer_path:
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
            joblib.dump(self.vectorizer, self.vectorizer_path)

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            raise RuntimeError(
                "Embedding function belum di-fit. Jalankan ingest_data.py dulu "
                "sebelum melakukan query, atau pastikan vectorizer_path benar."
            )
        vectors = self.vectorizer.transform(input).toarray()
        return vectors.tolist()


class VectorStore:
    """
    Kelas pembungkus ChromaDB untuk dua koleksi utama:
    - sop_documents      : untuk RAG (CS Agent)
    - case_history        : untuk semantic case search (Maintenance & CS Agent)
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fns = {}  # cache per collection_name, vocab tiap koleksi berbeda

    def _get_embedding_fn(self, collection_name: str) -> TfidfEmbeddingFunction:
        if collection_name not in self._embedding_fns:
            vec_path = os.path.join(self.persist_dir, "_vectorizers", f"{collection_name}.joblib")
            self._embedding_fns[collection_name] = TfidfEmbeddingFunction(vectorizer_path=vec_path)
        return self._embedding_fns[collection_name]

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name, embedding_function=self._get_embedding_fn(name)
        )

    def index_documents(self, collection_name: str, ids: list[str],
                         documents: list[str], metadatas: list[dict] = None):
        """Fit embedding khusus collection ini pada seluruh corpus, lalu index."""
        embedding_fn = self._get_embedding_fn(collection_name)
        embedding_fn.fit(documents)
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas or [{} for _ in ids],
        )
        return collection

    def query(self, collection_name: str, query_text: str, n_results: int = 3):
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results
