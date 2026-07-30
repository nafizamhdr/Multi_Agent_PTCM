"""
embedding_smoke_test.py
Verifikasi pipeline SentenceTransformerEmbeddingFunction + integrasi ChromaDB
TANPA mengunduh model asli dari HuggingFace Hub (untuk environment tanpa akses
internet). Caranya: membangun model transformer kecil dengan bobot RANDOM secara
lokal, menyimpannya dalam format yang kompatibel dengan library sentence-transformers,
lalu memuatnya lewat path lokal (bukan nama model di HuggingFace Hub).

Hasil embedding pada smoke test ini TIDAK bermakna secara semantik (bobot acak),
tapi berhasil/tidaknya proses ini membuktikan apakah kode integrasi -- tokenisasi,
pooling, forward pass, hingga upsert ke ChromaDB dan query -- bebas dari kesalahan
implementasi sebelum dipakai dengan model sungguhan.

Usage:
    python tools/embedding_smoke_test.py
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertConfig, BertModel, AutoTokenizer


def build_tiny_local_sentence_transformer(target_dir: str):
    """
    Membangun model sentence-transformers minimal secara lokal:
    - BertModel kecil berbobot random sebagai transformer backbone
    - Tokenizer WordPiece dilatih dari data lokal (tidak download apapun)
    - modules.json + config Pooling supaya bisa dibaca sentence_transformers.SentenceTransformer
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from sentence_transformers import models as st_models, SentenceTransformer

    os.makedirs(target_dir, exist_ok=True)

    # 1) Tokenizer WordPiece kecil, dilatih dari data lokal proyek
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "customer_service", "complaint_tickets.csv",
    )
    import pandas as pd
    texts = pd.read_csv(data_path)["isi_komplain"].tolist()

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=2000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tok.train_from_iterator(texts, trainer=trainer)
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )

    # 2) BertModel kecil, bobot random (bukan download)
    config = BertConfig(
        vocab_size=fast_tok.vocab_size + 10, hidden_size=128,
        num_hidden_layers=2, num_attention_heads=4, intermediate_size=256,
        max_position_embeddings=256,
    )
    model = BertModel(config)

    transformer_dir = os.path.join(target_dir, "transformer_backbone")
    model.save_pretrained(transformer_dir)
    fast_tok.save_pretrained(transformer_dir)

    # 3) Rakit jadi SentenceTransformer (transformer + mean pooling)
    word_embedding_model = st_models.Transformer(transformer_dir, max_seq_length=64)
    pooling_model = st_models.Pooling(word_embedding_model.get_word_embedding_dimension())
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    st_model.save(target_dir)
    return target_dir


def main():
    tmp_dir = tempfile.mkdtemp(prefix="st_smoke_")
    try:
        print("[1/3] Membangun model sentence-transformers kecil secara lokal (bobot random)...")
        local_model_path = build_tiny_local_sentence_transformer(os.path.join(tmp_dir, "tiny_st_model"))
        print(f"      -> tersimpan di {local_model_path}")

        print("[2/3] Uji SentenceTransformerEmbeddingFunction...")
        from tools.vector_store import SentenceTransformerEmbeddingFunction
        embed_fn = SentenceTransformerEmbeddingFunction(model_name_or_path=local_model_path)
        sample_texts = ["Barang saya telat sampai", "Produk yang diterima cacat", "Saya ingin retur barang"]
        vectors = embed_fn(sample_texts)
        assert len(vectors) == 3, "Jumlah vektor harus sama dengan jumlah input teks"
        assert len(vectors[0]) > 0, "Dimensi embedding harus > 0"
        print(f"      -> berhasil, {len(vectors)} vektor dihasilkan, dimensi={len(vectors[0])}")

        print("[3/3] Uji integrasi penuh dengan ChromaDB (index + query)...")
        import chromadb
        test_chroma_dir = os.path.join(tmp_dir, "test_chroma")
        client = chromadb.PersistentClient(path=test_chroma_dir)
        collection = client.get_or_create_collection(name="smoke_test", embedding_function=embed_fn)
        collection.upsert(
            ids=["1", "2", "3"],
            documents=sample_texts,
            metadatas=[{"idx": i} for i in range(3)],
        )
        result = collection.query(query_texts=["barang telat"], n_results=1)
        assert len(result["documents"][0]) == 1, "Query harus mengembalikan 1 hasil"
        print(f"      -> berhasil, query mengembalikan: {result['documents'][0][0][:50]!r}")

        print("\n[SMOKE TEST PASSED] Pipeline sentence-transformers + ChromaDB terverifikasi bebas "
              "galat implementasi. Hasil embedding pada test ini TIDAK bermakna secara semantik "
              "(bobot model random) -- untuk embedding semantik sungguhan, set EMBEDDING_BACKEND="
              "sentence-transformers pada .env dan jalankan ingest_data.py di environment dengan "
              "akses internet ke HuggingFace Hub.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
