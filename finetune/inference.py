"""
inference.py
Load base model (IndoBERT) + LoRA adapter hasil fine-tuning untuk
mengklasifikasikan teks komplain baru ke salah satu dari 4 kategori.

Usage (CLI):
    python finetune/inference.py "Barang saya belum sampai padahal sudah 5 hari"
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_DIR = os.path.join(BASE_DIR, "finetune", "lora_adapter")
BASE_MODEL_NAME = "indobenchmark/indobert-base-p1"

_model = None
_tokenizer = None
_id2label = None


def _load():
    global _model, _tokenizer, _id2label
    if _model is not None:
        return

    if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
        raise FileNotFoundError(
            f"LoRA adapter belum ditemukan di {ADAPTER_DIR}. "
            "Jalankan 'python finetune/prepare_dataset.py' lalu 'python finetune/train_lora.py' dulu."
        )

    with open(os.path.join(ADAPTER_DIR, "label_map.json"), encoding="utf-8") as f:
        label_map = json.load(f)
    _id2label = {int(k): v for k, v in label_map["id2label"].items()}

    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=len(_id2label)
    )
    _model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    _model.eval()


def classify_complaint(text: str) -> dict:
    """Mengembalikan kategori prediksi beserta distribusi probabilitas tiap kelas."""
    _load()
    inputs = _tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()

    pred_id = int(torch.argmax(logits, dim=-1)[0])
    return {
        "predicted_category": _id2label[pred_id],
        "confidence": round(probs[pred_id], 4),
        "distribution": {_id2label[i]: round(p, 4) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Barang saya belum sampai padahal sudah 5 hari dari estimasi"
    result = classify_complaint(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
