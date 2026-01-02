#!/usr/bin/env python3
"""
Odoo Translation Model Training Script
=======================================
Обучава Qwen2.5 модел за превод на Odoo от английски на български.

Поддържа:
- RunPod, Vast.ai, Lambda Labs, Kaggle, Colab
- QLoRA за ефективно обучение
- Експорт към GGUF за Ollama

Използване:
  python train.py                          # Default settings
  python train.py --model qwen2.5:7b       # По-малък модел
  python train.py --epochs 5               # Повече епохи
  python train.py --output ./my_model      # Друга директория
"""

import os
import sys
import json
import argparse
from pathlib import Path

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

DEFAULT_CONFIG = {
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
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    
    # Paths
    "train_file": "training_data/chatml_train.jsonl",
    "val_file": "training_data/chatml_val.jsonl",
    "output_dir": "./odoo_model_output",
    "gguf_output": "./odoo_model_gguf",
    
    # GGUF
    "gguf_quantization": "q4_k_m",
}

# Алтернативни модели
MODELS = {
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


def install_dependencies():
    """Инсталира нужните пакети"""
    print("=" * 60)
    print("ИНСТАЛИРАНЕ НА ПАКЕТИ")
    print("=" * 60)
    
    os.system("pip install -q unsloth")
    os.system("pip install -q --no-deps trl peft accelerate bitsandbytes")
    os.system("pip install -q datasets transformers sentencepiece protobuf")
    
    print("✓ Пакетите са инсталирани")


def load_dataset(config):
    """Зарежда и подготвя dataset"""
    from datasets import Dataset
    
    print("\n" + "=" * 60)
    print("ЗАРЕЖДАНЕ НА ДАННИ")
    print("=" * 60)
    
    train_file = Path(config["train_file"])
    
    if not train_file.exists():
        print(f"✗ Файлът не е намерен: {train_file}")
        print("\nПровери дали training_data.zip е разархивиран:")
        print("  unzip training_data.zip")
        sys.exit(1)
    
    # Зареди JSONL
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✓ Заредени samples: {len(data)}")
    
    # Форматирай за ChatML
    def format_chatml(example):
        text = ""
        for msg in example.get('messages', []):
            role = msg.get('role', '')
            content = msg.get('content', '')
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return {"text": text}
    
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_chatml, remove_columns=dataset.column_names)
    
    print(f"✓ Dataset подготвен: {len(dataset)} samples")
    
    # Покажи примерен sample
    print("\nПример:")
    print("-" * 40)
    print(dataset[0]['text'][:500])
    print("-" * 40)
    
    return dataset


def setup_model(config):
    """Зарежда и конфигурира модела"""
    from unsloth import FastLanguageModel
    import torch
    
    print("\n" + "=" * 60)
    print("ЗАРЕЖДАНЕ НА МОДЕЛ")
    print("=" * 60)
    
    print(f"Модел: {config['base_model']}")
    print(f"Max sequence length: {config['max_seq_length']}")
    
    # Зареди с 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["max_seq_length"],
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    print("✓ Модел зареден")
    
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
    
    print(f"✓ LoRA добавен: r={config['lora_r']}, alpha={config['lora_alpha']}")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    return model, tokenizer


def train_model(model, tokenizer, dataset, config):
    """Обучава модела"""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ")
    print("=" * 60)
    
    effective_batch = config["batch_size"] * config["gradient_accumulation_steps"]
    total_steps = (len(dataset) // effective_batch) * config["epochs"]
    
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Total steps: ~{total_steps}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        
        # Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        # Optimizer
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Saving
        save_strategy="epoch",
        save_total_limit=2,
        
        # Memory
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        
        # Other
        seed=42,
        report_to="none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
        packing=True,
    )
    
    print("\nОбучението започва...")
    print("-" * 40)
    
    # Train!
    trainer.train()
    
    print("-" * 40)
    print("✓ Обучението приключи!")
    
    return trainer


def save_model(model, tokenizer, config):
    """Запазва модела"""
    print("\n" + "=" * 60)
    print("ЗАПАЗВАНЕ НА МОДЕЛ")
    print("=" * 60)
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Запази LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ LoRA adapters запазени в: {output_dir}")
    
    return output_dir


def export_gguf(model, tokenizer, config):
    """Експортира като GGUF за Ollama"""
    print("\n" + "=" * 60)
    print("ЕКСПОРТ КЪМ GGUF")
    print("=" * 60)
    
    gguf_dir = Path(config["gguf_output"])
    gguf_dir.mkdir(parents=True, exist_ok=True)
    
    quant = config["gguf_quantization"]
    print(f"Quantization: {quant}")
    print("Това може да отнеме 10-20 минути...")
    
    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=quant
        )
        
        # Намери GGUF файла
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            gguf_file = gguf_files[0]
            size_gb = gguf_file.stat().st_size / (1024**3)
            print(f"✓ GGUF файл: {gguf_file.name} ({size_gb:.2f} GB)")
            
            # Създай Modelfile
            create_modelfile(gguf_dir, gguf_file.name)
            
            return gguf_file
    except Exception as e:
        print(f"⚠ Грешка при GGUF експорт: {e}")
        print("  Можеш да конвертираш ръчно по-късно с llama.cpp")
        return None


def create_modelfile(output_dir, gguf_filename):
    """Създава Modelfile за Ollama"""
    
    modelfile_content = f'''# Odoo Bulgarian Translation Model
# Fine-tuned Qwen2.5 за превод на Odoo ERP

FROM ./{gguf_filename}

SYSTEM """Ти си експертен преводач за Odoo ERP система. Превеждаш от английски на български език.
Използвай установената Odoo терминология. Отговаряй САМО с превода, без обяснения."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
'''
    
    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✓ Modelfile създаден: {modelfile_path}")


def test_model(model, tokenizer, config):
    """Тества модела с примери"""
    from unsloth import FastLanguageModel
    
    print("\n" + "=" * 60)
    print("ТЕСТВАНЕ")
    print("=" * 60)
    
    FastLanguageModel.for_inference(model)
    
    test_cases = [
        ("Sales Order", "sale.order", "Продажби"),
        ("Purchase Order", "purchase.order", "Покупки"),
        ("Invoice", "account.move", "Счетоводство"),
        ("Warehouse", "stock.warehouse", "Склад"),
        ("Expected Date", "stock.picking", "Склад"),
        ("Unit Price", "sale.order.line", "Продажби"),
        ("The quantity must be positive.", None, None),
        ("Confirm", None, None),
    ]
    
    print("\nТестови преводи:")
    print("-" * 50)
    
    for source, model_name, domain in test_cases:
        # Построй prompt
        prompt = "<|im_start|>system\n"
        prompt += "Ти си преводач за Odoo ERP. Превеждай от английски на български.<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += f'Преведи: "{source}"'
        
        if model_name or domain:
            prompt += "\nКонтекст:"
            if model_name:
                prompt += f" Модел: {model_name}"
            if domain:
                prompt += f", Домейн: {domain}"
                
        prompt += "<|im_end|>\n<|im_start|>assistant\n"
        
        # Генерирай
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Извлечи отговора
        if "<|im_start|>assistant" in response:
            translation = response.split("<|im_start|>assistant")[-1]
            translation = translation.split("<|im_end|>")[0].strip()
        else:
            translation = response.split(prompt)[-1].strip()
        
        print(f"  EN: {source}")
        print(f"  BG: {translation}")
        print()


def print_final_instructions(config):
    """Показва финални инструкции"""
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    
    print(f"""
Файлове за сваляне:
  1. LoRA adapters: {config['output_dir']}/
  2. GGUF модел: {config['gguf_output']}/

За да използваш модела с Ollama:

  1. Свали папката {config['gguf_output']}/
  
  2. На твоята машина:
     cd {config['gguf_output']}
     ollama create odoo-translate-bg -f Modelfile
  
  3. Тествай:
     ollama run odoo-translate-bg "Преведи: Sales Order"

  4. Или с Docker:
     docker cp {config['gguf_output']} ollama-container:/models/
     docker exec ollama-container ollama create odoo-bg -f /models/Modelfile

Архивиране за сваляне:
  zip -r odoo_model.zip {config['gguf_output']}/
""")


def main():
    parser = argparse.ArgumentParser(
        description='Train Odoo Translation Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примери:
  python train.py                              # Default (Qwen2.5-14B)
  python train.py --model qwen2.5:7b           # По-малък модел
  python train.py --epochs 5 --lr 1e-4         # Custom training
  python train.py --train-file my_data.jsonl   # Custom data
        """
    )
    
    parser.add_argument('--model', '-m', default='qwen2.5:14b',
                       choices=list(MODELS.keys()),
                       help='Модел за обучение (default: qwen2.5:14b)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Брой епохи (default: 3)')
    parser.add_argument('--batch-size', '-b', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--train-file', default='training_data/chatml_train.jsonl',
                       help='Training data file')
    parser.add_argument('--output', '-o', default='./odoo_model_output',
                       help='Output directory for LoRA')
    parser.add_argument('--gguf-output', default='./odoo_model_gguf',
                       help='Output directory for GGUF')
    parser.add_argument('--skip-gguf', action='store_true',
                       help='Skip GGUF export')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip model testing')
    
    args = parser.parse_args()
    
    # Настрой конфигурацията
    config = DEFAULT_CONFIG.copy()
    config["base_model"] = MODELS.get(args.model, args.model)
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["train_file"] = args.train_file
    config["output_dir"] = args.output
    config["gguf_output"] = args.gguf_output
    
    print("=" * 60)
    print("  ODOO TRANSLATION MODEL TRAINING")
    print("=" * 60)
    print(f"\nКонфигурация:")
    print(f"  Модел: {config['base_model']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Training file: {config['train_file']}")
    
    # Инсталирай пакети
    install_dependencies()
    
    # Зареди данни
    dataset = load_dataset(config)
    
    # Зареди модел
    model, tokenizer = setup_model(config)
    
    # Обучи
    trainer = train_model(model, tokenizer, dataset, config)
    
    # Запази
    save_model(model, tokenizer, config)
    
    # Тествай
    if not args.skip_test:
        test_model(model, tokenizer, config)
    
    # GGUF експорт
    if not args.skip_gguf:
        export_gguf(model, tokenizer, config)
    
    # Финални инструкции
    print_final_instructions(config)


if __name__ == '__main__':
    main()
