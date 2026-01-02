# Cloud Training Quick Start

## Файлове за качване в cloud

```
training_data/
├── chatml_train.jsonl    # Training данни
├── chatml_val.jsonl      # Validation данни
└── modules_analysis.json # LLM анализ (за справка)

train.py                  # Training скрипт
```

## Архивиране

```bash
cd /opt/odoo/odoo_translation_analyzer
zip -r upload.zip training_data/ train.py
```

---

## RunPod

1. https://runpod.io → Deploy → GPU Pod
2. GPU: **RTX 4090** ($0.44/час) или **A100** ($1.64/час)
3. Template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
4. Disk: 50GB container, 100GB volume

```bash
cd /workspace
# Качи upload.zip през Jupyter
unzip upload.zip

python train.py --model qwen2.5:14b
# или за по-бързо:
python train.py --model qwen2.5:7b
```

---

## Vast.ai

1. https://vast.ai → Search
2. Филтри: GPU RTX 4090/3090, VRAM 24GB+, Disk 100GB+
3. Image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`

```bash
cd /workspace
# Качи upload.zip
unzip upload.zip

python train.py
```

---

## Lambda Labs

1. https://lambdalabs.com/cloud
2. Launch: A100 40GB ($1.10/час)

```bash
# SSH към машината
scp upload.zip ubuntu@<ip>:~/
ssh ubuntu@<ip>

unzip upload.zip
python train.py
```

---

## Kaggle (Безплатно)

1. https://kaggle.com → Code → New Notebook
2. Settings → Accelerator → **GPU T4 x2**
3. Settings → Internet → **On**
4. Add Data → Upload `training_data.zip`

```python
# Cell 1
!pip install -q unsloth
!pip install -q --no-deps trl peft accelerate bitsandbytes

# Cell 2
!unzip /kaggle/input/your-dataset/training_data.zip -d ./

# Cell 3 - използвай по-малък модел за T4
!python train.py --model qwen2.5:7b --batch-size 1
```

---

## Google Colab Pro+

1. https://colab.google → New Notebook
2. Runtime → Change runtime type → **A100**

```python
# Cell 1
!pip install -q unsloth
!pip install -q --no-deps trl peft accelerate bitsandbytes

# Cell 2
from google.colab import files
uploaded = files.upload()  # Качи upload.zip
!unzip upload.zip

# Cell 3
!python train.py
```

---

## Параметри на train.py

```bash
python train.py --help

Опции:
  --model, -m      Модел (qwen2.5:14b, qwen2.5:7b, qwen2.5:3b, llama3.1:8b)
  --epochs, -e     Брой епохи (default: 3)
  --batch-size, -b Batch size (default: 2)
  --lr             Learning rate (default: 2e-4)
  --output, -o     Output директория за LoRA
  --gguf-output    Output директория за GGUF
  --skip-gguf      Пропусни GGUF експорт
  --skip-test      Пропусни тестване
```

### Примери:

```bash
# Default - Qwen2.5-14B
python train.py

# По-малък модел (по-бързо, по-малко RAM)
python train.py --model qwen2.5:7b

# Повече епохи
python train.py --epochs 5

# Custom batch size за малко RAM
python train.py --batch-size 1 --model qwen2.5:7b
```

---

## Очаквано време

| GPU | Модел | 15K samples | 30K samples |
|-----|-------|-------------|-------------|
| RTX 4090 | qwen2.5:14b | ~2-3 часа | ~4-5 часа |
| RTX 4090 | qwen2.5:7b | ~1-2 часа | ~2-3 часа |
| A100 40GB | qwen2.5:14b | ~1-1.5 часа | ~2-3 часа |
| 2x T4 | qwen2.5:7b | ~4-6 часа | ~8-10 часа |

---

## След обучението

1. **Архивирай GGUF:**
```bash
zip -r odoo_model.zip odoo_model_gguf/
# Свали през Jupyter/scp
```

2. **На твоята машина:**
```bash
unzip odoo_model.zip
cd odoo_model_gguf

# С Docker Ollama:
docker cp . ollama-amdgpu:/models/odoo/
docker exec -it ollama-amdgpu ollama create odoo-bg -f /models/odoo/Modelfile

# Тест:
docker exec -it ollama-amdgpu ollama run odoo-bg "Преведи: Sales Order"
```

---

## Troubleshooting

### Out of Memory
```bash
# Намали batch size
python train.py --batch-size 1 --model qwen2.5:7b
```

### GGUF експорт се проваля
```bash
# Пропусни GGUF, после конвертирай ръчно
python train.py --skip-gguf

# После с llama.cpp:
python convert.py odoo_model_output --outtype q4_k_m
```

### Бавно обучение
```bash
# Използвай по-малък модел
python train.py --model qwen2.5:7b
# или
python train.py --model qwen2.5:3b
```
