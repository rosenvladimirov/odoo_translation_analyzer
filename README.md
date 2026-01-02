# Odoo Translation Training Data Pipeline

Инструмент за анализ на Odoo source code и генериране на training data за обучение на модел за преводи с контекст.

## Концепция

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Odoo Source    │────▶│  Code Analysis   │────▶│  Semantic       │
│  - Python       │     │  - AST parsing   │     │  Context        │
│  - XML views    │     │  - Model extract │     │  - Domain       │
│  - PO files     │     │  - Field extract │     │  - Data type    │
└─────────────────┘     └──────────────────┘     │  - Workflow     │
                                                 │  - Relations    │
                                                 └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Training Data   │◀─────────────┘
                        │  - JSONL         │
                        │  - Alpaca        │
                        │  - ChatML        │
                        │  - Llama format  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Fine-tuned      │
                        │  Translation     │
                        │  Model           │
                        └──────────────────┘
```

## Инсталация

```bash
git clone <repo>
cd odoo_translation_analyzer
pip install -r requirements.txt  # (ако има)
```

## Използване

### 1. Анализ на Odoo source code

```bash
# Анализирай целия Odoo
python main.py analyze /path/to/odoo

# Анализирай само определени модули
python main.py analyze /path/to/odoo --modules sale,purchase,account,stock

# Запиши резултатите в директория
python main.py analyze /path/to/odoo --output ./analysis
```

### 2. Генериране на training data

```bash
# Генерирай всички формати
python main.py generate /path/to/odoo --output ./training_data

# Само Alpaca формат
python main.py generate /path/to/odoo --format alpaca

# За конкретни модули
python main.py generate /path/to/odoo --modules sale,account --output ./sale_account_data
```

### 3. Семантичен анализ

```bash
python main.py semantic /path/to/odoo --output ./semantic
```

### 4. Демо превод (генериране на prompt)

```bash
python main.py translate "Sales Order" --model sale.order --field name --domain Продажби

python main.py translate "Expected Date" --model purchase.order --field date_planned --type Datetime --few-shot
```

### 5. Информация за модел

```bash
python main.py info /path/to/odoo --model sale.order
```

## Структура на training data

### Simple JSONL формат

```json
{
  "source": "Order Date",
  "target": "Дата на поръчка",
  "model": "sale.order",
  "field": "date_order",
  "type": "field_label",
  "domain": "Продажби",
  "context": {
    "field_type": "Datetime",
    "help": "Date on which the order is confirmed",
    "required": true
  }
}
```

### Alpaca формат

```json
{
  "instruction": "Преведи следния текст от Odoo ERP на български език. Домейн: Продажби. Това е етикет на поле в модел sale.order.",
  "input": "Order Date",
  "output": "Дата на поръчка"
}
```

### ChatML формат

```json
{
  "messages": [
    {"role": "system", "content": "Ти си експертен преводач за Odoo ERP система..."},
    {"role": "user", "content": "Преведи: \"Order Date\"\nКонтекст: Модел: sale.order, Поле: date_order, Домейн: Продажби"},
    {"role": "assistant", "content": "Дата на поръчка"}
  ]
}
```

## Компоненти

### analyzer.py
- **OdooFieldVisitor** - AST visitor за извличане на Odoo полета и методи
- **OdooXMLAnalyzer** - Парсер на XML views, actions, menus
- **OdooPOAnalyzer** - Парсер на PO файлове с преводи
- **OdooSourceAnalyzer** - Главен анализатор

### semantic_analyzer.py
- **SemanticContext** - Семантичен контекст за превод
- **SemanticAnalyzer** - Дълбок анализ на бизнес логика
- **TranslationTerminology** - База с терминология

### training_generator.py
- **TrainingSample** - Един training sample с методи за различни формати
- **TrainingDataGenerator** - Генератор на training data
- **InferencePromptBuilder** - Генератор на prompts за inference

## Типове контекст

### Business Domain
- Продажби (sale.*)
- Покупки (purchase.*)
- Счетоводство (account.*)
- Склад (stock.*)
- HR (hr.*)
- Производство (mrp.*)
- CRM (crm.*)
- Проекти (project.*)

### Translation Types
- `field_label` - Етикет на поле
- `help_text` - Помощен текст
- `selection_value` - Стойност от dropdown
- `error_message` - Съобщение за грешка
- `button_label` - Етикет на бутон
- `menu_item` - Елемент от меню
- `model_description` - Описание на модел

### Data Types
- `date` - Дата/време
- `money` - Парични стойности
- `quantity` - Количества
- `reference` - Референции/кодове
- `status` - Статуси/състояния
- `percentage` - Проценти
- `contact` - Контактна информация

## Обучение на модел

### С Hugging Face transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

# Зареди данните
dataset = load_dataset('json', data_files='training_data/alpaca_train.json')

# Зареди модел (напр. Mistral, Llama)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Fine-tune
# ... training code ...
```

### С axolotl

```yaml
base_model: mistralai/Mistral-7B-v0.1
datasets:
  - path: training_data/alpaca_train.json
    type: alpaca
```

### С LLaMA-Factory

```bash
llamafactory-cli train \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset training_data/alpaca_train.json \
    --template alpaca
```

## Inference

```python
from training_generator import InferencePromptBuilder

builder = InferencePromptBuilder()

prompt = builder.build_prompt(
    source="Delivery Date",
    model_name="stock.picking",
    field_name="scheduled_date",
    field_type="Datetime",
    business_domain="Склад"
)

# Подай на модела
response = model.generate(prompt)
```

## Разширяване

### Добавяне на нови домейни

```python
# В semantic_analyzer.py
DOMAIN_MAPPING = {
    ...
    'my.custom.model': ('Моят домейн', 'Описание на модела'),
}
```

### Добавяне на терминология

```python
# В semantic_analyzer.py
STANDARD_ODOO_TERMS_BG = {
    ...
    'My Term': 'Моят термин',
}
```

---

# Cloud Training - Пълно ръководство

## Подготовка на данните

```bash
cd /opt/odoo/odoo_translation_analyzer

# Провери колко samples имаш
wc -l training_data/*.jsonl

# Архивирай за качване
zip -r upload.zip training_data/ train.py

# Провери размера (трябва да е 5-50 MB)
ls -lh upload.zip
```

---

## Vast.ai - Подробно ръководство

### Защо Vast.ai?
- **Най-евтино** - RTX 4090 от $0.25-0.50/час
- Голям избор от GPU-та
- Почасово плащане

### Стъпка 1: Регистрация

1. Отиди на https://vast.ai
2. Натисни **Sign Up**
3. Въведи email и парола
4. Потвърди email-а

### Стъпка 2: Добави Credits

1. Влез в акаунта
2. Отиди на **Billing** (горе вдясно)
3. Натисни **Add Credit**
4. Добави $5-10 (достатъчно за няколко обучения)
5. Плати с карта или crypto

### Стъпка 3: Намери машина

1. Отиди на **Search** (или **Console** → **Create**)
2. Настрой филтри:
   ```
   GPU Type: RTX 4090 или RTX 3090 или A6000
   GPU RAM: 24 GB+
   Disk Space: 100 GB+
   Download Speed: 100+ Mbps
   ```
3. Сортирай по **$/hr** (най-евтините първо)
4. Избери машина с добър **reliability score** (>95%)

### Стъпка 4: Launch Instance

1. Натисни **RENT** на избраната машина
2. Настройки:
   ```
   Docker Image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
   Disk Space: 100 GB
   ```
3. Натисни **RENT**
4. Изчакай да стартира (1-3 минути)

### Стъпка 5: Свържи се

1. Когато статусът стане **Running**, натисни **Open**
2. Ще се отвори Jupyter Lab
3. Или използвай SSH:
   ```bash
   # Копирай SSH командата от интерфейса
   ssh -p <port> root@<host>
   ```

### Стъпка 6: Качи файловете

**Вариант A - Jupyter Upload:**
1. В Jupyter отляво има файлов браузър
2. Натисни бутона за Upload (стрелка нагоре)
3. Избери `upload.zip`
4. Изчакай качването

**Вариант B - wget от URL:**
```bash
# Ако имаш файла качен някъде
cd /workspace
wget https://your-server.com/upload.zip
```

**Вариант C - SCP:**
```bash
# От твоята машина
scp -P <port> upload.zip root@<host>:/workspace/
```

### Стъпка 7: Стартирай обучението

В Jupyter Terminal или SSH:

```bash
cd /workspace

# Разархивирай
unzip upload.zip

# Инсталирай dependencies
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes

# Провери CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Стартирай обучението
python train.py --model qwen2.5:14b

# За по-малък модел (по-бързо):
python train.py --model qwen2.5:7b
```

### Стъпка 8: Изчакай обучението

```
============================================================
ОБУЧЕНИЕ
============================================================
Samples: 15000
Epochs: 3
Effective batch size: 16
Total steps: ~2800

Обучението започва...
------------------------------------------
Step 10/2800 | Loss: 2.45 | LR: 0.0001
Step 20/2800 | Loss: 1.89 | LR: 0.0002
...
```

**Очаквано време:**
| GPU | 15K samples | 30K samples |
|-----|-------------|-------------|
| RTX 4090 | 2-3 часа | 4-5 часа |
| RTX 3090 | 3-4 часа | 6-7 часа |
| A6000 | 2-3 часа | 4-5 часа |

### Стъпка 9: Свали резултата

След приключване:

```bash
cd /workspace

# Архивирай GGUF модела
zip -r odoo_model.zip odoo_model_gguf/

# Виж размера
ls -lh odoo_model.zip
# Очаквано: 8-10 GB за 14B модел, 4-5 GB за 7B модел
```

**Сваляне:**
- В Jupyter: десен клик на `odoo_model.zip` → Download
- Или SCP от твоята машина:
  ```bash
  scp -P <port> root@<host>:/workspace/odoo_model.zip ./
  ```

### Стъпка 10: СПРИ МАШИНАТА!

**ВАЖНО:** Vast.ai таксува докато машината работи!

1. В интерфейса натисни **Destroy** на инстанцията
2. Потвърди

### Vast.ai Troubleshooting

**Грешка: Out of Memory**
```bash
# Използвай по-малък batch size
python train.py --batch-size 1 --model qwen2.5:7b
```

**Грешка: Connection lost**
```bash
# Пусни в screen сесия
screen -S training
python train.py
# Ctrl+A, D за да излезеш
# screen -r training за да се върнеш
```

**Грешка: CUDA not available**
```bash
# Провери дали GPU е разпознат
nvidia-smi
# Ако не показва нищо - рестартирай инстанцията
```

---

## RunPod - Подробно ръководство

### Защо RunPod?
- **По-стабилен** от Vast.ai
- По-лесен интерфейс
- Готови templates

### Стъпка 1: Регистрация

1. Отиди на https://runpod.io
2. Натисни **Sign Up**
3. Регистрирай се с email или GitHub
4. Потвърди акаунта

### Стъпка 2: Добави Credits

1. Влез в акаунта
2. Натисни на **$0.00** горе вдясно
3. **Add Credits**
4. Добави $5-10
5. Плати с карта

### Стъпка 3: Създай Pod

1. Отиди на **Pods** → **+ Deploy**
2. Избери GPU:
   ```
   Препоръчани:
   - RTX 4090 ($0.44/час) - най-добър избор
   - RTX A6000 ($0.79/час) - повече VRAM
   - A100 40GB ($1.64/час) - най-бърз
   ```
3. Избери Template:
   ```
   RunPod Pytorch 2.1
   или
   runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
   ```
4. Storage:
   ```
   Container Disk: 50 GB
   Volume Disk: 100 GB (persistent)
   ```
5. Натисни **Deploy**

### Стъпка 4: Изчакай стартиране

- Статусът ще мине през: **Pending** → **Building** → **Running**
- Обикновено отнема 1-5 минути

### Стъпка 5: Отвори терминал

1. Когато статусът е **Running**, натисни **Connect**
2. Избери **Jupyter Lab** (препоръчително) или **SSH**
3. Ще се отвори в нов таб

### Стъпка 6: Качи файловете

**В Jupyter Lab:**
1. Отляво има файлов браузър
2. Навигирай до `/workspace/`
3. Натисни Upload бутона (стрелка нагоре)
4. Избери `upload.zip`
5. Изчакай (progress bar)

**Алтернативно с wget:**
```bash
cd /workspace
# Ако имаш файла на публичен URL
wget https://your-url.com/upload.zip
```

### Стъпка 7: Подготви средата

Отвори Terminal в Jupyter:

```bash
cd /workspace

# Разархивирай
unzip upload.zip

# Инсталирай пакети
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes

# Провери GPU
nvidia-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Стъпка 8: Стартирай обучението

```bash
cd /workspace

# С 14B модел (по-добър, но по-бавен)
python train.py --model qwen2.5:14b

# С 7B модел (по-бърз)
python train.py --model qwen2.5:7b

# С custom параметри
python train.py --model qwen2.5:14b --epochs 5 --batch-size 1
```

### Стъпка 9: Мониторинг

Докато работи ще виждаш:

```
============================================================
ОБУЧЕНИЕ
============================================================
Samples: 15234
Epochs: 3
Batch size: 2
Gradient accumulation: 8
Effective batch size: 16
Total steps: ~2855
Learning rate: 0.0002

Обучението започва...
------------------------------------------
{'loss': 2.4521, 'learning_rate': 0.0001, 'epoch': 0.1}
{'loss': 1.8934, 'learning_rate': 0.0002, 'epoch': 0.2}
...
```

### Стъпка 10: Свали модела

След успешно обучение:

```bash
cd /workspace

# Провери какво е генерирано
ls -la odoo_model_gguf/

# Архивирай
zip -r odoo_model.zip odoo_model_gguf/

# Размер
ls -lh odoo_model.zip
```

**Сваляне от Jupyter:**
1. В файловия браузър намери `odoo_model.zip`
2. Десен клик → **Download**

**Или с runpodctl:**
```bash
# На твоята машина инсталирай runpodctl
pip install runpodctl

# Свали файла
runpodctl receive <pod_id>:/workspace/odoo_model.zip ./
```

### Стъпка 11: Спри Pod-а

**ВАЖНО:** Таксуването продължава докато Pod-ът работи!

1. В RunPod интерфейса намери твоя Pod
2. Натисни **Stop** (ако искаш да го запазиш за по-късно)
3. Или **Terminate** (за да изтриеш напълно)

### RunPod Troubleshooting

**Pod не стартира:**
- Опитай друг GPU тип
- Провери дали има наличност

**Out of Memory:**
```bash
python train.py --batch-size 1 --model qwen2.5:7b
```

**Обучението прекъсна:**
```bash
# Пусни в screen
screen -S train
python train.py
# Ctrl+A, D
```

**Jupyter не се отваря:**
- Изчакай 2-3 минути
- Refresh страницата
- Опитай SSH вместо Jupyter

---

## След обучението - Локално използване

### 1. Разархивирай на твоята машина

```bash
cd ~
unzip odoo_model.zip
cd odoo_model_gguf
ls -la
```

### 2. Импортирай в Ollama

**Ако Ollama е в Docker:**

```bash
# Копирай файловете в контейнера
docker cp odoo_model_gguf/. ollama-amdgpu:/models/odoo/

# Влез в контейнера
docker exec -it ollama-amdgpu bash

# Създай модела
cd /models/odoo
ollama create odoo-translate-bg -f Modelfile

# Излез
exit
```

**Ако Ollama е локален:**

```bash
cd odoo_model_gguf
ollama create odoo-translate-bg -f Modelfile
```

### 3. Тествай модела

```bash
# Интерактивен режим
docker exec -it ollama-amdgpu ollama run odoo-translate-bg

# В prompt-а:
>>> Преведи: "Sales Order"
>>> Контекст: Модел: sale.order, Домейн: Продажби
```

### 4. API използване

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "odoo-translate-bg",
  "prompt": "Преведи: \"Expected Delivery Date\"\nКонтекст: Модел: stock.picking, Домейн: Склад",
  "stream": false
}'
```

### 5. Python интеграция

```python
import requests

def translate_odoo(text, model=None, domain=None):
    prompt = f'Преведи: "{text}"'
    if model or domain:
        prompt += "\nКонтекст:"
        if model:
            prompt += f" Модел: {model}"
        if domain:
            prompt += f", Домейн: {domain}"
    
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'odoo-translate-bg',
            'prompt': prompt,
            'stream': False
        }
    )
    return response.json()['response'].strip()

# Използване
print(translate_odoo("Sales Order", "sale.order", "Продажби"))
# -> Поръчка за продажба

print(translate_odoo("Invoice", "account.move", "Счетоводство"))
# -> Фактура
```

---

## Сравнение на платформите

| Критерий | Vast.ai | RunPod |
|----------|---------|--------|
| **Цена RTX 4090** | $0.25-0.50/час | $0.44/час |
| **Стабилност** | Променлива | По-добра |
| **Интерфейс** | По-сложен | По-лесен |
| **Наличност** | Голям избор | По-малък избор |
| **Templates** | Няма | Има готови |
| **Минимален депозит** | $5 | $5 |
| **Плащане** | Карта/Crypto | Карта |

**Препоръка:**
- **За начинаещи:** RunPod (по-лесен интерфейс)
- **За спестяване:** Vast.ai (по-евтино)
- **За стабилност:** RunPod или Lambda Labs

---

## Цялостен workflow

```
1. ПОДГОТОВКА (локално)
   └── python batch_analyzer.py
   └── zip -r upload.zip training_data/ train.py

2. CLOUD TRAINING (Vast.ai/RunPod)
   └── Качи upload.zip
   └── pip install unsloth ...
   └── python train.py
   └── zip -r odoo_model.zip odoo_model_gguf/
   └── Свали odoo_model.zip

3. ЛОКАЛНО ИЗПОЛЗВАНЕ
   └── unzip odoo_model.zip
   └── ollama create odoo-translate-bg -f Modelfile
   └── ollama run odoo-translate-bg
```

---

## TODO

- [ ] Извличане на error messages от ValidationError
- [ ] Анализ на JavaScript файлове
- [ ] Извличане на report templates
- [ ] Поддръжка на OWL components (Odoo 17+)
- [ ] Web UI за преглед на анализа
- [ ] Интеграция с Odoo Studio за real-time превод
