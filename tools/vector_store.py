"""
vector_store.py
Wrapper ChromaDB untuk keperluan RAG (dokumen SOP) dan semantic case search
(histori maintenance & komplain CS).

Mendukung DUA embedding backend, dipilih lewat environment variable
EMBEDDING_BACKEND (lihat .env.example):

- "tfidf" (default)              : scikit-learn TF-IDF, ringan, 100% offline,
                                    tidak butuh download apapun.
- "sentence-transformers"        : model transformer multibahasa pretrained
                                    (default: paraphrase-multilingual-MiniLM-L12-v2),
                                    menangkap makna semantik jauh lebih baik
                                    daripada TF-IDF, tapi butuh akses internet
                                    untuk mengunduh bobot model saat pertama
                                    kali dijalankan.

Kalau EMBEDDING_BACKEND="sentence-transformers" tapi model gagal dimuat
(misal tidak ada akses internet), VectorStore otomatis fallback ke TF-IDF
dengan peringatan di console -- sistem tetap bisa jalan tanpa crash.
"""

import os
import joblib
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function custom untuk ChromaDB berbasis TF-IDF.
    Di-fit sekali saat koleksi dibuat, lalu dipakai untuk transform query baru.
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


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function berbasis model transformer pretrained (sentence-transformers).
    Tidak perlu di-fit (model sudah pretrained) -- method fit() disediakan hanya
    supaya interface-nya konsisten dengan TfidfEmbeddingFunction (VectorStore
    memanggil .fit() secara seragam untuk kedua backend).

    model_name_or_path: nama model di HuggingFace Hub (butuh internet saat pertama
    kali dijalankan, lalu ter-cache lokal), ATAU path lokal ke model yang sudah
    disimpan sebelumnya (dipakai untuk smoke test offline, lihat
    tools/embedding_smoke_test.py).
    """

    def __init__(self, model_name_or_path: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name_or_path)

    def fit(self, corpus: list[str]):
        pass  # no-op: model sudah pretrained, tidak butuh fitting pada corpus lokal

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self.model.encode(list(input), convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


class VectorStore:
    """
    Kelas pembungkus ChromaDB untuk dua koleksi utama:
    - sop_documents      : untuk RAG (CS Agent)
    - case_history        : untuk semantic case search (Maintenance & CS Agent)
    """

    def __init__(self, persist_dir: str = "./chroma_db", embedding_backend: str = None,
                 st_model_name: str = None):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fns = {}  # cache per collection_name, vocab/model tiap koleksi
        self.embedding_backend = embedding_backend or os.getenv("EMBEDDING_BACKEND", "tfidf")
        self.st_model_name = st_model_name or os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )

    def _get_embedding_fn(self, collection_name: str):
        if collection_name in self._embedding_fns:
            return self._embedding_fns[collection_name]

        backend = self.embedding_backend
        if backend == "sentence-transformers":
            try:
                fn = SentenceTransformerEmbeddingFunction(self.st_model_name)
                print(f"[VectorStore] '{collection_name}': embedding backend = "
                      f"sentence-transformers ({self.st_model_name})")
                self._embedding_fns[collection_name] = fn
                return fn
            except Exception as e:
                print(f"[VectorStore] WARNING: gagal memuat sentence-transformers "
                      f"({type(e).__name__}: {str(e)[:150]}). Fallback ke TF-IDF untuk "
                      f"koleksi '{collection_name}'.")
                backend = "tfidf"

        vec_path = os.path.join(self.persist_dir, "_vectorizers", f"{collection_name}.joblib")
        fn = TfidfEmbeddingFunction(vectorizer_path=vec_path)
        print(f"[VectorStore] '{collection_name}': embedding backend = tfidf")
        self._embedding_fns[collection_name] = fn
        return fn

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
