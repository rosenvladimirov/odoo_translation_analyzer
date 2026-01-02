#!/bin/bash
# ==============================================
# Анализ на локални Odoo инсталации
# ==============================================
#
# Използване:
#   ./run_local.sh
#
# Анализира:
#   /opt/odoo/base/odoo-18.0 (Community)
#   /opt/odoo/ee/odoo-18.0 (Enterprise)  
#   /opt/odoo/oca/odoo-18.0 (OCA)
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  ODOO TRANSLATION TRAINING DATA"
echo "=================================================="
echo ""

# Провери пътища
echo "Проверка на източници..."
for path in "/opt/odoo/base/odoo-18.0" "/opt/odoo/ee/odoo-18.0" "/opt/odoo/oca/odoo-18.0"; do
    if [ -d "$path" ]; then
        echo "  ✓ $path"
    else
        echo "  ✗ $path (не съществува)"
    fi
done
echo ""

# Стартирай анализа
echo "Стартиране на анализа..."
python3 analyze_local.py

# Покажи резултата
echo ""
echo "=================================================="
echo "Генерирани файлове:"
echo "=================================================="
ls -lh training_data/*.jsonl 2>/dev/null || echo "Няма JSONL файлове"
ls -lh training_data/*.json 2>/dev/null | head -5

echo ""
echo "Брой samples:"
wc -l training_data/*.jsonl 2>/dev/null || true

echo ""
echo "=================================================="
echo "За да качиш данните в cloud:"
echo "=================================================="
echo "  zip -r training_data.zip training_data/"
echo "  # Качи training_data.zip в RunPod"
echo ""
