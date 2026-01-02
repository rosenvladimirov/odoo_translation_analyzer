#!/bin/bash
# RunPod Quick Start Script
# Копирай този скрипт в терминала на RunPod

set -e

echo "=========================================="
echo "  ODOO TRANSLATION MODEL - QUICK START"
echo "=========================================="

# 1. Setup
echo "[1/6] Инсталиране на пакети..."
pip install -q unsloth
pip install -q --no-deps trl peft accelerate bitsandbytes
pip install -q datasets transformers sentencepiece protobuf

# 2. Създай работна директория
echo "[2/6] Създаване на работна директория..."
cd /workspace
mkdir -p odoo_training
cd odoo_training

# 3. Провери за training data
echo "[3/6] Проверка за training data..."
if [ ! -f "training_data/chatml_train.jsonl" ]; then
    echo ""
    echo "⚠️  Training data не е намерен!"
    echo ""
    echo "Моля качи training data по един от следните начини:"
    echo ""
    echo "Опция A - През Jupyter:"
    echo "  1. Отвори Jupyter Lab"
    echo "  2. Навигирай до /workspace/odoo_training/"
    echo "  3. Създай папка 'training_data'"
    echo "  4. Качи chatml_train.jsonl файла"
    echo ""
    echo "Опция B - През wget (ако имаш URL):"
    echo "  mkdir -p training_data"
    echo "  wget YOUR_URL -O training_data/chatml_train.jsonl"
    echo ""
    echo "Опция C - Примерни данни за тест:"
    echo "  Стартирай: bash runpod_start.sh --demo"
    echo ""
    
    # Ако е --demo flag, създай примерни данни
    if [ "$1" == "--demo" ]; then
        echo "Създаване на демо данни..."
        mkdir -p training_data
        cat > training_data/chatml_train.jsonl << 'EOF'
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Sales Order\"\nКонтекст: Модел: sale.order"},{"role":"assistant","content":"Поръчка за продажба"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Purchase Order\"\nКонтекст: Модел: purchase.order"},{"role":"assistant","content":"Поръчка за покупка"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Invoice\"\nКонтекст: Модел: account.move"},{"role":"assistant","content":"Фактура"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Customer\"\nКонтекст: Модел: res.partner"},{"role":"assistant","content":"Клиент"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Warehouse\"\nКонтекст: Модел: stock.warehouse"},{"role":"assistant","content":"Склад"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Product\"\nКонтекст: Модел: product.product"},{"role":"assistant","content":"Продукт"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Quantity\"\nКонтекст: Модел: stock.move"},{"role":"assistant","content":"Количество"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Unit Price\"\nКонтекст: Модел: sale.order.line"},{"role":"assistant","content":"Единична цена"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Total\"\nКонтекст: Модел: sale.order"},{"role":"assistant","content":"Общо"}]}
{"messages":[{"role":"system","content":"Ти си преводач за Odoo ERP. Превеждай от английски на български."},{"role":"user","content":"Преведи: \"Confirm\"\nКонтекст: Тип: button_label"},{"role":"assistant","content":"Потвърди"}]}
EOF
        echo "✓ Демо данни създадени (10 примера)"
    else
        exit 1
    fi
fi

# 4. Създай training скрипт
echo "[4/6] Създаване на training скрипт..."
cat > train.py << 'TRAINSCRIPT'
import os
import json
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Config
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_R = 64
LORA_ALPHA = 128
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4

print("=" * 60)
print("Loading model...")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

data = []
with open("training_data/chatml_train.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

def format_chat(example):
    text = ""
    for msg in example['messages']:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

dataset = Dataset.from_list(data).map(format_chat)
print(f"Dataset: {len(dataset)} samples")

print("=" * 60)
print("Training...")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir="./output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
    ),
    packing=True,
)

trainer.train()

print("=" * 60)
print("Saving GGUF...")
print("=" * 60)

model.save_pretrained_gguf("./gguf_output", tokenizer, quantization_method="q4_k_m")

# Create Modelfile
with open("./gguf_output/Modelfile", "w") as f:
    f.write('''FROM ./unsloth.Q4_K_M.gguf

SYSTEM """Ти си преводач за Odoo ERP. Превеждай от английски на български."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
''')

print("=" * 60)
print("DONE!")
print("=" * 60)
print("\nDownload: ./gguf_output/")
print("Then run: ollama create odoo-translate-bg -f Modelfile")
TRAINSCRIPT

# 5. Покажи GPU info
echo "[5/6] GPU информация..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')"

# 6. Стартирай
echo ""
echo "[6/6] Готово за обучение!"
echo ""
echo "=========================================="
echo "  За да започнеш обучението, изпълни:"
echo "  python train.py"
echo "=========================================="
echo ""
echo "След обучението, свали папката gguf_output/"
echo "и стартирай в Ollama:"
echo "  ollama create odoo-translate-bg -f Modelfile"
