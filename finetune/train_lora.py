"""
train_lora.py
Fine-tuning IndoBERT (indobenchmark/indobert-base-p1) menggunakan LoRA (Low-Rank
Adaptation) via library PEFT, untuk tugas klasifikasi kategori tiket komplain
pelanggan (4 kelas: keterlambatan_pengiriman, produk_cacat, retur_barang,
eskalasi_kritikal).

Kenapa LoRA (bukan full fine-tuning)?
LoRA hanya melatih matriks rank-rendah tambahan yang disisipkan ke layer
attention (query, value projection), sehingga hanya <1% dari total parameter
model yang diperbarui. Ini jauh lebih hemat komputasi dan memori dibanding
full fine-tuning, sekaligus tetap merupakan teknik fine-tuning yang sah
(bukan hanya training-from-scratch atau classifier head biasa).

Kenapa IndoBERT?
IndoBERT adalah model bahasa berbasis BERT yang di-pretrain khusus pada
korpus Bahasa Indonesia, sehingga representasi bahasanya jauh lebih relevan
untuk teks komplain pelanggan berbahasa Indonesia dibandingkan BERT
multilingual atau model bahasa Inggris.

Prasyarat:
    python finetune/prepare_dataset.py   (sekali saja)

Usage:
    python finetune/train_lora.py
    python finetune/train_lora.py --smoke-test   (uji pipeline tanpa download bobot pretrained,
                                                    lihat catatan SMOKE TEST di bawah)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "finetune", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "finetune", "lora_adapter")
BASE_MODEL_NAME = "indobenchmark/indobert-base-p1"


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    with open(os.path.join(DATA_DIR, "label_map.json"), encoding="utf-8") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]

    train_df["labels"] = train_df["label"].map(label2id)
    val_df["labels"] = val_df["label"].map(label2id)

    return Dataset.from_pandas(train_df[["text", "labels"]]), Dataset.from_pandas(val_df[["text", "labels"]]), label_map


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1}


def build_model(num_labels: int, smoke_test: bool):
    """
    smoke_test=True: bangun model dengan ARSITEKTUR yang sama tapi bobot RANDOM
    (tanpa download dari HuggingFace Hub), hanya untuk memverifikasi pipeline
    training/LoRA berjalan tanpa error di environment tanpa akses internet ke
    HuggingFace. Hasil klasifikasinya TIDAK bermakna pada mode ini -- untuk
    hasil sungguhan, jalankan tanpa flag --smoke-test di environment dengan
    akses internet.
    """
    if smoke_test:
        from transformers import BertConfig
        config = BertConfig(
            vocab_size=3000, hidden_size=256, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=512,
            max_position_embeddings=512, num_labels=num_labels,
        )
        model = AutoModelForSequenceClassification.from_config(config)
        tokenizer = _dummy_tokenizer()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels)
    return model, tokenizer


def _has_internet() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
        return True
    except Exception:
        return False


def _dummy_tokenizer():
    """Tokenizer WordPiece minimal untuk smoke test murni offline (tanpa download apapun)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=3000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    tok.train_from_iterator(train_df["text"].tolist(), trainer=trainer)
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    return fast_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Verifikasi pipeline tanpa download bobot pretrained (bobot random)")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    train_ds, val_ds, label_map = load_data()
    num_labels = len(label_map["label2id"])

    model, tokenizer = build_model(num_labels, args.smoke_test)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # ---- LoRA config ----
    # Menyisipkan adapter rank-rendah pada matriks query & value di setiap layer
    # attention BERT. r=8 adalah rank adapter (semakin kecil = semakin hemat
    # parameter, r=8 adalah nilai umum untuk task klasifikasi teks berukuran kecil).
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    print("=== Parameter yang dilatih (LoRA) vs total parameter model ===")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "finetune", "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Evaluasi akhir pada data validasi ===")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] LoRA adapter tersimpan di {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
