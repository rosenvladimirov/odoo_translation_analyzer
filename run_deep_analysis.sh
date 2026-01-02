#!/bin/bash
# ==============================================
# Deep Semantic Analyzer Runner
# ==============================================
# Анализира Odoo код с помощта на локален LLM
# за да генерира training data с богат контекст
#
# Изисква: Ollama с модел (qwen2.5:14b препоръчително)
#
# Използване:
#   ./run_deep_analysis.sh
#   ./run_deep_analysis.sh --model qwen2.5:7b
#   ./run_deep_analysis.sh --model deepseek-coder:6.7b
# ==============================================

set -e

# Defaults
MODEL="qwen2.5:14b"
OUTPUT="./training_data"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--model MODEL] [--output DIR]"
            echo ""
            echo "Options:"
            echo "  --model, -m   Ollama модел (default: qwen2.5:14b)"
            echo "  --output, -o  Изходна директория (default: ./training_data)"
            echo ""
            echo "Препоръчани модели:"
            echo "  qwen2.5:14b       - Най-добър за анализ (нужни ~10GB RAM)"
            echo "  qwen2.5:7b        - По-бърз, по-малко RAM"
            echo "  deepseek-coder:6.7b - Специализиран за код"
            echo "  codellama:13b     - Добър за код анализ"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "  DEEP SEMANTIC ODOO ANALYZER"
echo "=================================================="
echo ""
echo "Модел: $MODEL"
echo "Output: $OUTPUT"
echo ""

# Провери Ollama API (работи и с Docker)
echo "Проверка на Ollama API..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✗ Ollama API не е достъпен на localhost:11434"
    echo ""
    echo "Ако използваш Docker:"
    echo "  docker ps | grep ollama"
    echo "  docker logs ollama-amdgpu"
    echo ""
    echo "Ако е локален:"
    echo "  ollama serve"
    exit 1
fi

# Покажи налични модели
echo "✓ Ollama API работи"
MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print(' '.join([m['name'] for m in json.load(sys.stdin).get('models',[])]))" 2>/dev/null)
echo "  Налични модели: $MODELS"

# Провери модела
echo "Проверка на модел $MODEL..."
if ! echo "$MODELS" | grep -q "${MODEL%%:*}"; then
    echo "⚠ Модел $MODEL не е намерен."
    echo "  Свали го с:"
    echo "    docker exec -it ollama-amdgpu ollama pull $MODEL"
    echo "  или локално:"
    echo "    ollama pull $MODEL"
    exit 1
fi
echo "✓ Модел готов"

# Провери пътища
echo ""
echo "Проверка на Odoo директории..."
for path in "/opt/odoo/base/odoo-18.0" "/opt/odoo/ee/odoo-18.0" "/opt/odoo/oca/odoo-18.0"; do
    if [ -d "$path" ]; then
        count=$(find "$path" -name "*.py" -type f 2>/dev/null | wc -l)
        echo "  ✓ $path ($count .py файла)"
    else
        echo "  ✗ $path (не съществува)"
    fi
done

# Стартирай анализа
echo ""
echo "=================================================="
echo "Стартиране на дълбок анализ..."
echo "Това може да отнеме 30-60 минути!"
echo "=================================================="
echo ""

python3 deep_analyzer.py --model "$MODEL" --output "$OUTPUT"

# Резултати
echo ""
echo "=================================================="
echo "РЕЗУЛТАТИ"
echo "=================================================="
if [ -d "$OUTPUT" ]; then
    echo "Генерирани файлове:"
    ls -lh "$OUTPUT"/*.jsonl 2>/dev/null || true
    
    echo ""
    echo "Брой samples:"
    wc -l "$OUTPUT"/*.jsonl 2>/dev/null || true
fi

echo ""
echo "=================================================="
echo "СЛЕДВАЩИ СТЪПКИ"
echo "=================================================="
echo "1. Архивирай: zip -r training_data.zip $OUTPUT/"
echo "2. Качи в RunPod"
echo "3. Стартирай обучение"
echo ""
