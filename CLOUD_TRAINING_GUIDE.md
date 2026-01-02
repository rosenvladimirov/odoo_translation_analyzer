# Cloud Training Guide за Odoo Translation Model

## Стъпка по стъпка инструкции

---

## Опция 1: RunPod (Препоръчително)

### 1. Регистрация и setup

1. Отиди на https://runpod.io
2. Регистрирай се и добави $5-10 credits (достатъчно за няколко обучения)

### 2. Създай Pod

1. **Templates** → **GPU Pods** → **Deploy**
2. Избери GPU:
   - **RTX 4090** ($0.44/час) - препоръчително
   - **A100 40GB** ($1.64/час) - по-бързо
3. **Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
4. **Container Disk**: 50GB
5. **Volume Disk**: 100GB (за модела)
6. Натисни **Deploy**

### 3. Качи файловете

Когато Pod-ът стартира, отвори **Jupyter Lab** или **SSH**.

```bash
# В терминала на Pod-а

# Клонирай/качи training скриптовете
cd /workspace
git clone https://github.com/YOUR_REPO/odoo_translation_analyzer.git
# ИЛИ качи файловете през Jupyter

# Качи training data
# Можеш да го направиш през Jupyter upload или:
wget YOUR_TRAINING_DATA_URL -O training_data.zip
unzip training_data.zip
```

### 4. Стартирай обучението

```bash
cd /workspace/odoo_translation_analyzer

# Инсталирай dependencies (автоматично в скрипта, но можеш и ръчно)
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes

# Редактирай CONFIG в cloud_training.py ако трябва
nano cloud_training.py

# Стартирай!
python cloud_training.py
```

### 5. Свали резултата

След обучението ще имаш:
- `/workspace/qwen_odoo_bg/` - LoRA adapters
- `/workspace/qwen_odoo_bg_gguf/` - GGUF файл за Ollama

```bash
# Архивирай GGUF файла
cd /workspace
zip -r qwen_odoo_bg_gguf.zip qwen_odoo_bg_gguf/

# Свали през Jupyter или с runpodctl
```

### 6. Спри Pod-а!
**ВАЖНО**: След като свалиш файловете, спри Pod-а за да не плащаш!

---

## Опция 2: Vast.ai (По-евтино)

### 1. Регистрация

1. https://vast.ai
2. Добави credits ($5-10)

### 2. Намери машина

1. **Search** → Филтри:
   - GPU: RTX 4090 или A6000
   - VRAM: 24GB+
   - Disk: 100GB+
2. Сортирай по цена
3. Избери най-евтиния с добра връзка (>100 Mbps)

### 3. Launch с Docker image

```
Image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
Disk: 100GB
```

### 4. Същите стъпки като RunPod

---

## Опция 3: Google Colab Pro+ (Най-лесно, но лимитирано)

### 1. Colab Notebook

```python
# Първа клетка - setup
!pip install -q unsloth
!pip install -q --no-deps trl peft accelerate bitsandbytes

# Качи training data
from google.colab import files
uploaded = files.upload()  # Качи training_data.zip

!unzip training_data.zip

# Копирай cloud_training.py съдържанието и го стартирай
```

### Лимитации на Colab:
- Максимум ~8 часа непрекъснато
- Може да прекъсне сесията
- A100 не винаги е налична

---

## Опция 4: Kaggle (Безплатно!)

### 1. Създай Notebook

1. https://kaggle.com → **Code** → **New Notebook**
2. **Settings** → **Accelerator** → **GPU T4 x2**
3. **Settings** → **Internet** → **On**

### 2. Notebook код

```python
# Клетка 1: Setup
!pip install -q unsloth
!pip install -q --no-deps trl peft accelerate bitsandbytes

# Клетка 2: Качи данни
# Качи training_data.zip като Kaggle Dataset или директно

# Клетка 3: Training
# ... cloud_training.py код ...
```

### Лимитации:
- 2x T4 (по-бавно от 4090)
- 30 часа/седмица лимит
- Сесия до 12 часа

---

## Локално използване след обучението

### 1. Инсталирай Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Mac
brew install ollama

# Windows
# Свали от https://ollama.com/download
```

### 2. Импортирай модела

```bash
# Разархивирай сваления файл
unzip qwen_odoo_bg_gguf.zip
cd qwen_odoo_bg_gguf

# Създай модел в Ollama
ollama create odoo-translate-bg -f Modelfile

# Провери
ollama list
```

### 3. Използвай модела

```bash
# Интерактивен режим
ollama run odoo-translate-bg

# Prompt
>>> Преведи: "Sales Order"
>>> Контекст: Модел: sale.order, Домейн: Продажби
```

### 4. API достъп

```bash
# REST API
curl http://localhost:11434/api/generate -d '{
  "model": "odoo-translate-bg",
  "prompt": "Преведи: \"Delivery Date\"\nКонтекст: Модел: stock.picking",
  "stream": false
}'
```

### 5. Python интеграция

```python
import ollama

def translate_odoo(text, model=None, field=None, domain=None):
    prompt = f'Преведи: "{text}"'
    if model or field or domain:
        context = []
        if model: context.append(f"Модел: {model}")
        if field: context.append(f"Поле: {field}")
        if domain: context.append(f"Домейн: {domain}")
        prompt += "\nКонтекст: " + ", ".join(context)
    
    response = ollama.generate(
        model='odoo-translate-bg',
        prompt=prompt
    )
    return response['response'].strip()

# Използване
print(translate_odoo("Sales Order", model="sale.order", domain="Продажби"))
# -> Поръчка за продажба
```

---

## Очаквани резултати

### Размер на файловете
- LoRA adapters: ~200-500 MB
- GGUF Q4_K_M: ~8-9 GB
- GGUF Q8_0: ~15-16 GB

### Време за обучение (10K samples)
| GPU | Време | Цена |
|-----|-------|------|
| RTX 4090 | ~1.5-2 часа | ~$1 |
| A100 40GB | ~45-60 мин | ~$1.50 |
| 2x T4 (Kaggle) | ~4-6 часа | Безплатно |

### Хардуер за inference
| GGUF Quantization | VRAM нужен | Качество |
|-------------------|------------|----------|
| Q4_K_M | ~10 GB | Добро |
| Q5_K_M | ~12 GB | По-добро |
| Q8_0 | ~18 GB | Най-добро |

---

## Troubleshooting

### Out of Memory при обучение
```python
# Намали batch size
CONFIG["batch_size"] = 1
CONFIG["gradient_accumulation_steps"] = 16
```

### Модел не се зарежда в Ollama
```bash
# Провери GGUF файла
ls -la qwen_odoo_bg_gguf/*.gguf

# Пробвай да създадеш отново
ollama rm odoo-translate-bg
ollama create odoo-translate-bg -f Modelfile
```

### Лоши преводи
1. Провери качеството на training data
2. Увеличи epochs на 4-5
3. Добави повече примери за проблемните случаи

---

## Checklist преди обучение

- [ ] Training data готов (`training_data/chatml_train.jsonl`)
- [ ] Достатъчно credits в cloud платформата
- [ ] GPU с 24GB+ VRAM избран
- [ ] Disk space: минимум 80GB свободни
- [ ] Backup план ако сесията прекъсне
