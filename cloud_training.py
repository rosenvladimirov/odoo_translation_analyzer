#!/usr/bin/env python3
"""
Cloud Training Script за Qwen2.5:14B Odoo Translation Model
============================================================
Използвай в RunPod, Vast.ai, Lambda Labs или Kaggle

Изисквания: RTX 4090 / A100 / A6000 с 24GB+ VRAM
Време: 1-3 часа за 10-20K samples
Цена: ~$2-5 в RunPod

Изход: GGUF файл за Ollama
"""

import os
import json
import torch
from pathlib import Path

# ============================================
# КОНФИГУРАЦИЯ - ПРОМЕНИ СПОРЕД НУЖДИТЕ
# ============================================

CONFIG = {
    # Модел
    "base_model": "Qwen/Qwen2.5-14B-Instruct",
    "max_seq_length": 2048,
    
    # LoRA параметри
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    
    # Training
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,  # Ефективен batch = 16
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    
    # Paths
    "dataset_path": "training_data/chatml_train.jsonl",
    "output_dir": "./qwen_odoo_bg",
    "gguf_output": "./qwen_odoo_bg_gguf",
    
    # GGUF quantization за Ollama
    "gguf_quantization": "q4_k_m",  # Баланс размер/качество
}


def install_dependencies():
    """Инсталира нужните пакети"""
    os.system("pip install -q unsloth")
    os.system("pip install -q --no-deps trl peft accelerate bitsandbytes")
    os.system("pip install -q datasets transformers sentencepiece protobuf")


def prepare_dataset(config):
    """Подготвя dataset в правилен формат"""
    from datasets import load_dataset, Dataset
    
    # Зареди JSONL файла
    if config["dataset_path"].endswith('.jsonl'):
        data = []
        with open(config["dataset_path"], 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset('json', data_files=config["dataset_path"])['train']
    
    # Форматиране за ChatML
    def format_chat(example):
        if 'messages' in example:
            # Вече е в ChatML формат
            messages = example['messages']
        else:
            # Конвертирай от прост формат
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": build_user_prompt(example)},
                {"role": "assistant", "content": example.get('target', example.get('output', ''))}
            ]
        
        # Форматирай като текст
        text = ""
        for msg in messages:
            if msg["role"] == "system":
                text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
        return {"text": text}
    
    dataset = dataset.map(format_chat)
    
    print(f"Dataset готов: {len(dataset)} примера")
    print(f"Примерен запис:\n{dataset[0]['text'][:500]}...")
    
    return dataset


def get_system_prompt():
    """System prompt за модела"""
    return """Ти си експертен преводач за Odoo ERP система. Превеждаш от английски на български език.

Правила:
1. Използвай установената Odoo терминология на български
2. Съобразявай се с контекста (модел, тип поле, бизнес домейн)
3. Преводът трябва да е естествен и професионален
4. Отговаряй САМО с превода, без обяснения

Стандартна терминология:
- Sales Order = Поръчка за продажба
- Purchase Order = Поръчка за покупка  
- Invoice = Фактура
- Warehouse = Склад
- Customer = Клиент
- Vendor/Supplier = Доставчик
- Product = Продукт
- Quantity = Количество
- Unit Price = Единична цена
- Total = Общо
- Tax = Данък
- Payment = Плащане
- Delivery = Доставка
- Confirm = Потвърди
- Cancel = Отмени
- Draft = Чернова
- Posted = Осчетоводено"""


def build_user_prompt(example):
    """Генерира user prompt от данните"""
    source = example.get('source', example.get('input', ''))
    
    parts = [f'Преведи: "{source}"']
    
    context_parts = []
    if example.get('model'):
        context_parts.append(f"Модел: {example['model']}")
    if example.get('field'):
        context_parts.append(f"Поле: {example['field']}")
    if example.get('domain'):
        context_parts.append(f"Домейн: {example['domain']}")
    if example.get('type'):
        context_parts.append(f"Тип: {example['type']}")
        
    if context_parts:
        parts.append("Контекст: " + ", ".join(context_parts))
        
    return '\n'.join(parts)


def train(config):
    """Основна training функция"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    print("=" * 60)
    print("ЗАРЕЖДАНЕ НА МОДЕЛ")
    print("=" * 60)
    
    # Зареди модел с 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["max_seq_length"],
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    print(f"Модел зареден: {config['base_model']}")
    
    # Добави LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print(f"LoRA добавен: r={config['lora_r']}, alpha={config['lora_alpha']}")
    
    # Подготви dataset
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА НА ДАННИ")
    print("=" * 60)
    
    dataset = prepare_dataset(config)
    
    # Training arguments
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ")
    print("=" * 60)
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        
        # Оптимизации
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        
        # Logging
        logging_steps=10,
        save_strategy="epoch",
        
        # Memory
        gradient_checkpointing=True,
        max_grad_norm=0.3,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
        packing=True,  # По-ефективно използване на sequence length
    )
    
    # Покажи статистики
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_stats.name}")
    print(f"VRAM: {gpu_stats.total_memory / 1024**3:.1f} GB")
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {config['epochs']}")
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    # Train!
    trainer.train()
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕТО ПРИКЛЮЧИ!")
    print("=" * 60)
    
    return model, tokenizer


def export_gguf(model, tokenizer, config):
    """Експортира модела като GGUF за Ollama"""
    from unsloth import FastLanguageModel
    
    print("\n" + "=" * 60)
    print("ЕКСПОРТИРАНЕ КАТО GGUF")
    print("=" * 60)
    
    # Запази LoRA адаптерите
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"LoRA adapters запазени в: {config['output_dir']}")
    
    # Експортирай като GGUF
    print(f"\nКонвертиране към GGUF ({config['gguf_quantization']})...")
    
    model.save_pretrained_gguf(
        config["gguf_output"],
        tokenizer,
        quantization_method=config["gguf_quantization"]
    )
    
    # Намери GGUF файла
    gguf_files = list(Path(config["gguf_output"]).glob("*.gguf"))
    if gguf_files:
        gguf_path = gguf_files[0]
        size_gb = gguf_path.stat().st_size / (1024**3)
        print(f"\nGGUF файл създаден: {gguf_path}")
        print(f"Размер: {size_gb:.2f} GB")
        
        # Създай Modelfile за Ollama
        create_modelfile(config["gguf_output"], gguf_path.name)
    
    return gguf_files[0] if gguf_files else None


def create_modelfile(output_dir, gguf_filename):
    """Създава Modelfile за Ollama"""
    
    modelfile_content = f'''# Odoo Bulgarian Translation Model
# Базиран на Qwen2.5:14B, fine-tuned за превод на Odoo

FROM ./{gguf_filename}

# System prompt
SYSTEM """Ти си експертен преводач за Odoo ERP система. Превеждаш от английски на български език.
Използвай установената Odoo терминология. Отговаряй САМО с превода."""

# Параметри за по-консистентни преводи
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Template за Qwen
TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
'''
    
    modelfile_path = Path(output_dir) / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"\nModelfile създаден: {modelfile_path}")
    print("\nЗа да използваш модела в Ollama:")
    print(f"  cd {output_dir}")
    print(f"  ollama create odoo-translate-bg -f Modelfile")
    print(f"  ollama run odoo-translate-bg")


def test_model(model, tokenizer):
    """Тества модела с няколко примера"""
    from unsloth import FastLanguageModel
    
    print("\n" + "=" * 60)
    print("ТЕСТВАНЕ НА МОДЕЛА")
    print("=" * 60)
    
    FastLanguageModel.for_inference(model)
    
    test_cases = [
        ("Sales Order", "sale.order", "Продажби"),
        ("Expected Date", "purchase.order", "Покупки"),
        ("Warehouse", "stock.warehouse", "Склад"),
        ("Post Entries", "account.move", "Счетоводство"),
        ("The quantity must be positive.", None, None),
    ]
    
    for source, model_name, domain in test_cases:
        prompt = f"<|im_start|>system\n{get_system_prompt()}<|im_end|>\n"
        
        user_content = f'Преведи: "{source}"'
        if model_name:
            user_content += f"\nКонтекст: Модел: {model_name}"
        if domain:
            user_content += f", Домейн: {domain}"
            
        prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Извлечи само assistant отговора
        if "<|im_start|>assistant" in response:
            translation = response.split("<|im_start|>assistant")[-1]
            translation = translation.split("<|im_end|>")[0].strip()
        else:
            translation = response
            
        print(f"\n  EN: {source}")
        print(f"  BG: {translation}")


def main():
    """Главна функция"""
    print("=" * 60)
    print("QWEN2.5:14B ODOO TRANSLATION TRAINING")
    print("=" * 60)
    
    # Провери CUDA
    if not torch.cuda.is_available():
        print("ГРЕШКА: CUDA не е налична!")
        return
        
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Инсталирай dependencies
    print("\nИнсталиране на пакети...")
    install_dependencies()
    
    # Провери dataset
    if not Path(CONFIG["dataset_path"]).exists():
        print(f"\nГРЕШКА: Dataset не е намерен: {CONFIG['dataset_path']}")
        print("Първо генерирай training data с:")
        print("  python main.py generate /path/to/odoo --output training_data")
        return
    
    # Train
    model, tokenizer = train(CONFIG)
    
    # Test
    test_model(model, tokenizer)
    
    # Export GGUF
    gguf_path = export_gguf(model, tokenizer, CONFIG)
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"\nФайлове за изтегляне:")
    print(f"  1. LoRA adapters: {CONFIG['output_dir']}/")
    print(f"  2. GGUF модел: {CONFIG['gguf_output']}/")
    print(f"\nСвали GGUF файла и Modelfile, после:")
    print(f"  ollama create odoo-translate-bg -f Modelfile")


if __name__ == "__main__":
    main()
