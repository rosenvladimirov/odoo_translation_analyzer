#!/bin/bash
# ==============================================
# Odoo Translation Training Data Collection
# ==============================================
# Този скрипт сваля, анализира и генерира 
# training данни от Odoo source code
#
# Използване:
#   ./collect_all.sh              # Пълен процес
#   ./collect_all.sh --quick      # Само Odoo 18
#   ./collect_all.sh --no-download # Без сваляне
# ==============================================

set -e

# Конфигурация
WORK_DIR="./odoo_sources"
OUTPUT_DIR="./training_data"
QUICK_MODE=false
SKIP_DOWNLOAD=false

# Парсване на аргументи
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --no-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Непознат аргумент: $1"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "  ODOO TRANSLATION TRAINING DATA COLLECTION"
echo "=================================================="
echo ""
echo "Работна директория: $WORK_DIR"
echo "Изходна директория: $OUTPUT_DIR"
echo "Quick mode: $QUICK_MODE"
echo ""

# Създай директории
mkdir -p "$WORK_DIR"
mkdir -p "$OUTPUT_DIR"

# ==============================================
# СТЪПКА 1: Сваляне на source code
# ==============================================

if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "=================================================="
    echo "СТЪПКА 1: Сваляне на Odoo source code"
    echo "=================================================="
    
    cd "$WORK_DIR"
    
    # Функция за клониране
    clone_repo() {
        local name=$1
        local url=$2
        local branch=$3
        
        if [ -d "$name" ]; then
            echo "✓ $name вече съществува"
        else
            echo "⬇ Сваляне на $name ($branch)..."
            git clone --depth 1 --branch "$branch" "$url" "$name" 2>/dev/null || {
                echo "  ⚠ Неуспешно сваляне на $name"
                return 1
            }
            echo "  ✓ $name свален"
        fi
    }
    
    # Odoo Community
    echo ""
    echo "--- Odoo Community ---"
    clone_repo "odoo-18" "https://github.com/odoo/odoo.git" "18.0"
    
    if [ "$QUICK_MODE" = false ]; then
        clone_repo "odoo-17" "https://github.com/odoo/odoo.git" "17.0"
        clone_repo "odoo-16" "https://github.com/odoo/odoo.git" "16.0"
        
        # OCA модули
        echo ""
        echo "--- OCA Модули ---"
        clone_repo "oca-l10n-bulgaria" "https://github.com/OCA/l10n-bulgaria.git" "16.0" || true
        clone_repo "oca-account" "https://github.com/OCA/account-financial-tools.git" "16.0" || true
        clone_repo "oca-sale" "https://github.com/OCA/sale-workflow.git" "16.0" || true
        clone_repo "oca-purchase" "https://github.com/OCA/purchase-workflow.git" "16.0" || true
        clone_repo "oca-stock" "https://github.com/OCA/stock-logistics-workflow.git" "16.0" || true
        clone_repo "oca-hr" "https://github.com/OCA/hr.git" "16.0" || true
        clone_repo "oca-manufacture" "https://github.com/OCA/manufacture.git" "16.0" || true
        clone_repo "oca-project" "https://github.com/OCA/project.git" "16.0" || true
        clone_repo "oca-crm" "https://github.com/OCA/crm.git" "16.0" || true
        clone_repo "oca-reporting" "https://github.com/OCA/reporting-engine.git" "16.0" || true
        clone_repo "oca-server-tools" "https://github.com/OCA/server-tools.git" "16.0" || true
    fi
    
    cd ..
fi

# ==============================================
# СТЪПКА 2: Събиране на PO преводи
# ==============================================

echo ""
echo "=================================================="
echo "СТЪПКА 2: Събиране на български преводи от PO файлове"
echo "=================================================="

# Събери всички български PO файлове
echo "Търсене на български преводи..."

BG_PO_DIR="$OUTPUT_DIR/bg_translations"
mkdir -p "$BG_PO_DIR"

find "$WORK_DIR" -name "bg.po" -o -name "bg_BG.po" 2>/dev/null | while read po_file; do
    # Копирай с уникално име
    module_name=$(basename $(dirname $(dirname "$po_file")))
    source_name=$(basename $(dirname $(dirname $(dirname "$po_file"))))
    cp "$po_file" "$BG_PO_DIR/${source_name}_${module_name}_bg.po" 2>/dev/null || true
done

PO_COUNT=$(ls -1 "$BG_PO_DIR"/*.po 2>/dev/null | wc -l)
echo "✓ Намерени $PO_COUNT български PO файла"

# ==============================================
# СТЪПКА 3: Извличане на преводи от PO
# ==============================================

echo ""
echo "=================================================="
echo "СТЪПКА 3: Парсване на преводи"
echo "=================================================="

# Създай Python скрипт за извличане
cat > "$OUTPUT_DIR/extract_po.py" << 'PYEOF'
#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

def parse_po_file(filepath):
    """Парсва PO файл и връща dict с преводи"""
    translations = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return {}
    
    # Regex за msgid/msgstr двойки
    pattern = r'msgid\s+"(.+?)"\s*\nmsgstr\s+"(.+?)"'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        source = match.group(1).replace('\\n', '\n').replace('\\"', '"')
        target = match.group(2).replace('\\n', '\n').replace('\\"', '"')
        
        # Филтрирай празни и твърде дълги
        if source and target and len(source) < 500 and len(target) < 500:
            # Премахни escapes
            source = source.strip()
            target = target.strip()
            if source and target and source != target:
                translations[source] = target
                
    return translations

def main():
    po_dir = Path('bg_translations')
    all_translations = {}
    
    for po_file in po_dir.glob('*.po'):
        trans = parse_po_file(po_file)
        all_translations.update(trans)
        print(f"  {po_file.name}: {len(trans)} превода")
    
    # Запази като JSON
    with open('bg_translations.json', 'w', encoding='utf-8') as f:
        json.dump(all_translations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Общо уникални преводи: {len(all_translations)}")
    
    # Покажи примери
    print("\nПримери:")
    for source, target in list(all_translations.items())[:10]:
        print(f"  {source[:50]} → {target[:50]}")

if __name__ == '__main__':
    main()
PYEOF

cd "$OUTPUT_DIR"
python3 extract_po.py
cd ..

# ==============================================
# СТЪПКА 4: Анализ на Python код
# ==============================================

echo ""
echo "=================================================="
echo "СТЪПКА 4: Анализ на Odoo модели и полета"
echo "=================================================="

# Анализирай всеки източник
for source_dir in "$WORK_DIR"/*/; do
    source_name=$(basename "$source_dir")
    echo "Анализиране на $source_name..."
    
    # Използвай нашия analyzer
    python3 -c "
import sys
sys.path.insert(0, '.')
from analyzer import OdooSourceAnalyzer
import json

analyzer = OdooSourceAnalyzer('$source_dir')
try:
    analyzer.analyze()
    
    # Експортирай
    output = {
        'summary': analyzer.generate_summary(),
        'models': {}
    }
    
    for name, model in analyzer.models.items():
        output['models'][name] = {
            'description': model.description,
            'module': model.module,
            'business_domain': model.business_domain,
            'fields': {
                fname: {
                    'string': f.string,
                    'help': f.help,
                    'field_type': f.field_type
                }
                for fname, f in model.fields.items()
                if f.string
            }
        }
    
    with open('$OUTPUT_DIR/${source_name}_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    print(f'  ✓ {len(output[\"models\"])} модела')
except Exception as e:
    print(f'  ⚠ Грешка: {e}')
" 2>/dev/null || echo "  ⚠ Пропуснат"
done

# ==============================================
# СТЪПКА 5: Генериране на Training Data
# ==============================================

echo ""
echo "=================================================="
echo "СТЪПКА 5: Генериране на training samples"
echo "=================================================="

cat > "$OUTPUT_DIR/generate_training.py" << 'PYEOF'
#!/usr/bin/env python3
import json
from pathlib import Path
import hashlib

def get_system_prompt():
    return """Ти си експертен преводач за Odoo ERP система. Превеждаш от английски на български език.

Правила:
1. Използвай установената Odoo терминология на български
2. Съобразявай се с контекста (модел, тип поле, бизнес домейн)
3. Преводът трябва да е естествен и професионален
4. Отговаряй САМО с превода, без обяснения"""

def create_sample(source, target, model='', field='', domain='', sample_type='translation'):
    """Създава ChatML sample"""
    
    user_content = f'Преведи: "{source}"'
    
    context_parts = []
    if model:
        context_parts.append(f"Модел: {model}")
    if field:
        context_parts.append(f"Поле: {field}")
    if domain:
        context_parts.append(f"Домейн: {domain}")
    if sample_type and sample_type != 'translation':
        type_names = {
            'field_label': 'Етикет на поле',
            'help_text': 'Помощен текст',
            'selection_value': 'Стойност от списък',
        }
        context_parts.append(f"Тип: {type_names.get(sample_type, sample_type)}")
        
    if context_parts:
        user_content += "\nКонтекст: " + ", ".join(context_parts)
    
    return {
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target}
        ]
    }

def main():
    output_dir = Path('.')
    all_samples = []
    seen = set()
    
    # 1. Зареди преводи от PO файлове
    print("Зареждане на PO преводи...")
    try:
        with open('bg_translations.json', 'r', encoding='utf-8') as f:
            po_translations = json.load(f)
        print(f"  ✓ {len(po_translations)} превода от PO")
    except:
        po_translations = {}
        print("  ⚠ Няма PO преводи")
    
    # 2. Зареди анализирани модели
    print("Зареждане на анализирани модели...")
    all_models = {}
    
    for analysis_file in output_dir.glob('*_analysis.json'):
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_models.update(data.get('models', {}))
        except:
            pass
    
    print(f"  ✓ {len(all_models)} модела")
    
    # 3. Генерирай samples от модели с преводи
    print("Генериране на samples...")
    
    for model_name, model in all_models.items():
        domain = model.get('business_domain', '')
        
        for field_name, field in model.get('fields', {}).items():
            source = field.get('string')
            if not source:
                continue
                
            # Търси превод
            target = po_translations.get(source)
            if not target:
                continue
                
            # Уникален ключ
            key = hashlib.md5(f"{source}|{target}|{model_name}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            
            # Създай sample
            sample = create_sample(
                source=source,
                target=target,
                model=model_name,
                field=field_name,
                domain=domain,
                sample_type='field_label'
            )
            all_samples.append(sample)
            
            # Вариация без контекст
            key2 = hashlib.md5(f"{source}|{target}|nocontext".encode()).hexdigest()
            if key2 not in seen:
                seen.add(key2)
                all_samples.append(create_sample(source, target))
    
    # 4. Добави останалите преводи от PO
    for source, target in po_translations.items():
        key = hashlib.md5(f"{source}|{target}|po".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            all_samples.append(create_sample(source, target))
    
    print(f"  ✓ {len(all_samples)} samples генерирани")
    
    # 5. Split train/val
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # 6. Запази
    with open('chatml_train.jsonl', 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
    with open('chatml_val.jsonl', 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Записани:")
    print(f"  chatml_train.jsonl: {len(train_samples)} samples")
    print(f"  chatml_val.jsonl: {len(val_samples)} samples")
    
    # Статистики
    print(f"\nСтатистики:")
    print(f"  Общо samples: {len(all_samples)}")
    print(f"  Training: {len(train_samples)} (90%)")
    print(f"  Validation: {len(val_samples)} (10%)")

if __name__ == '__main__':
    main()
PYEOF

cd "$OUTPUT_DIR"
python3 generate_training.py
cd ..

# ==============================================
# ФИНАЛНО ОБОБЩЕНИЕ
# ==============================================

echo ""
echo "=================================================="
echo "✓ ГОТОВО!"
echo "=================================================="
echo ""
echo "Генерирани файлове в $OUTPUT_DIR/:"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (няма JSONL файлове)"
echo ""
echo "Следващи стъпки:"
echo "  1. Качи $OUTPUT_DIR/ в RunPod/Vast.ai"
echo "  2. Стартирай cloud_training.py"
echo "  3. Свали GGUF модела"
echo "  4. ollama create odoo-translate-bg -f Modelfile"
echo ""
