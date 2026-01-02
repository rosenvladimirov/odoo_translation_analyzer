#!/usr/bin/env python3
"""
Анализира съществуващи Odoo инсталации и генерира training data
Специално за:
  - /opt/odoo/base/odoo-18.0 (Community)
  - /opt/odoo/ee/odoo-18.0 (Enterprise)
  - /opt/odoo/oca/odoo-18.0 (OCA модули)
"""

import os
import sys
import json
import re
import ast
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

ODOO_PATHS = [
    ("/opt/odoo/base/odoo-18.0", "community"),
    ("/opt/odoo/ee/odoo-18.0", "enterprise"),
    ("/opt/odoo/oca/odoo-18.0", "oca"),
]

OUTPUT_DIR = "./training_data"

# ============================================
# FIELD TYPES
# ============================================

FIELD_TYPES = {
    'Char', 'Text', 'Html', 'Integer', 'Float', 'Monetary',
    'Boolean', 'Date', 'Datetime', 'Binary', 'Image',
    'Selection', 'Many2one', 'One2many', 'Many2many',
    'Reference', 'Properties', 'PropertiesDefinition', 'Json'
}

# ============================================
# BUSINESS DOMAINS
# ============================================

DOMAIN_MAP = {
    'sale': 'Продажби',
    'purchase': 'Покупки',
    'account': 'Счетоводство',
    'stock': 'Склад',
    'mrp': 'Производство',
    'hr': 'Човешки ресурси',
    'project': 'Проекти',
    'crm': 'CRM',
    'website': 'Уебсайт',
    'pos': 'Точка на продажба',
    'fleet': 'Автопарк',
    'maintenance': 'Поддръжка',
    'helpdesk': 'Поддръжка',
    'mail': 'Съобщения',
    'calendar': 'Календар',
    'event': 'Събития',
    'survey': 'Анкети',
    'lunch': 'Обяд',
    'repair': 'Ремонти',
    'quality': 'Качество',
    'planning': 'Планиране',
    'timesheet': 'Отчитане на време',
    'expense': 'Разходи',
    'approvals': 'Одобрения',
    'documents': 'Документи',
    'sign': 'Подписване',
    'social': 'Социални мрежи',
    'marketing': 'Маркетинг',
    'mass_mailing': 'Масови имейли',
    'sms': 'SMS',
    'voip': 'VoIP',
    'iot': 'IoT',
    'barcode': 'Баркод',
    'l10n': 'Локализация',
}

# ============================================
# PO PARSER
# ============================================

def parse_po_file(filepath: str) -> dict:
    """Парсва PO файл"""
    translations = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return {}
    
    # Многоредови msgid/msgstr
    blocks = re.split(r'\n\n+', content)
    
    for block in blocks:
        # Извлечи msgid
        msgid_match = re.search(r'msgid\s+"((?:[^"\\]|\\.)*)"\s*(?:\n"((?:[^"\\]|\\.)*)")*', block)
        msgstr_match = re.search(r'msgstr\s+"((?:[^"\\]|\\.)*)"\s*(?:\n"((?:[^"\\]|\\.)*)")*', block)
        
        if msgid_match and msgstr_match:
            # Събери всички части
            msgid_parts = re.findall(r'"((?:[^"\\]|\\.)*)"', 
                                     block[block.find('msgid'):block.find('msgstr')])
            msgstr_parts = re.findall(r'"((?:[^"\\]|\\.)*)"', 
                                      block[block.find('msgstr'):])
            
            source = ''.join(msgid_parts)
            target = ''.join(msgstr_parts)
            
            # Unescape
            source = source.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            target = target.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            
            source = source.strip()
            target = target.strip()
            
            # Филтрирай
            if (source and target and 
                source != target and 
                len(source) >= 2 and 
                len(source) <= 500 and
                len(target) <= 500 and
                not source.startswith('#')):
                translations[source] = target
                
    return translations


def collect_po_translations(paths: list) -> dict:
    """Събира всички български преводи"""
    all_translations = {}
    po_files_found = 0
    
    print("\n" + "=" * 60)
    print("СЪБИРАНЕ НА ПРЕВОДИ ОТ PO ФАЙЛОВЕ")
    print("=" * 60)
    
    for base_path, source_type in paths:
        path = Path(base_path)
        if not path.exists():
            print(f"⚠ Път не съществува: {base_path}")
            continue
            
        # Намери всички български PO файлове
        for po_file in path.rglob("bg.po"):
            translations = parse_po_file(str(po_file))
            all_translations.update(translations)
            po_files_found += 1
            
        for po_file in path.rglob("bg_BG.po"):
            translations = parse_po_file(str(po_file))
            all_translations.update(translations)
            po_files_found += 1
            
    print(f"✓ Намерени PO файлове: {po_files_found}")
    print(f"✓ Уникални преводи: {len(all_translations)}")
    
    return all_translations


# ============================================
# PYTHON AST PARSER
# ============================================

@dataclass
class FieldInfo:
    name: str
    field_type: str
    string: Optional[str] = None
    help: Optional[str] = None
    selection: Optional[list] = None
    comodel_name: Optional[str] = None
    required: bool = False


@dataclass 
class ModelInfo:
    name: str
    description: Optional[str] = None
    module: str = ""
    business_domain: str = ""
    fields: dict = field(default_factory=dict)


class OdooASTVisitor(ast.NodeVisitor):
    """AST visitor за Odoo модели"""
    
    def __init__(self):
        self.models = []
        self.current_model = None
        
    def visit_ClassDef(self, node):
        """Обработва class дефиниция"""
        model_name = None
        model_inherit = []
        model_description = None
        fields = {}
        
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id == '_name':
                            model_name = self._get_string(item.value)
                        elif target.id == '_inherit':
                            model_inherit = self._get_list_or_string(item.value)
                        elif target.id == '_description':
                            model_description = self._get_string(item.value)
                        else:
                            # Може да е поле
                            field_info = self._parse_field(target.id, item.value)
                            if field_info:
                                fields[target.id] = field_info
                                
        # Определи име на модела
        final_name = model_name or (model_inherit[0] if model_inherit else None)
        
        if final_name and fields:
            self.models.append(ModelInfo(
                name=final_name,
                description=model_description,
                fields=fields
            ))
            
        self.generic_visit(node)
        
    def _parse_field(self, name: str, node) -> Optional[FieldInfo]:
        """Парсва дефиниция на поле"""
        if not isinstance(node, ast.Call):
            return None
            
        # Определи тип
        field_type = None
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FIELD_TYPES:
                field_type = node.func.attr
        elif isinstance(node.func, ast.Name):
            if node.func.id in FIELD_TYPES:
                field_type = node.func.id
                
        if not field_type:
            return None
            
        field_info = FieldInfo(name=name, field_type=field_type)
        
        for kw in node.keywords:
            if kw.arg == 'string':
                field_info.string = self._get_string(kw.value)
            elif kw.arg == 'help':
                field_info.help = self._get_string(kw.value)
            elif kw.arg == 'selection':
                field_info.selection = self._parse_selection(kw.value)
            elif kw.arg == 'comodel_name':
                field_info.comodel_name = self._get_string(kw.value)
            elif kw.arg == 'required':
                field_info.required = self._get_bool(kw.value)
                
        return field_info
        
    def _parse_selection(self, node) -> list:
        """Парсва selection стойности"""
        selections = []
        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) >= 2:
                    key = self._get_string(item.elts[0])
                    label = self._get_string(item.elts[1])
                    if key and label:
                        selections.append((key, label))
        return selections
        
    def _get_string(self, node) -> Optional[str]:
        if isinstance(node, ast.Constant):
            return str(node.value) if node.value else None
        elif isinstance(node, ast.Str):
            return node.s
        return None
        
    def _get_list_or_string(self, node) -> list:
        if isinstance(node, ast.Constant):
            return [node.value] if node.value else []
        elif isinstance(node, ast.Str):
            return [node.s]
        elif isinstance(node, ast.List):
            return [self._get_string(el) for el in node.elts if self._get_string(el)]
        return []
        
    def _get_bool(self, node) -> bool:
        if isinstance(node, ast.Constant):
            return bool(node.value)
        elif isinstance(node, ast.NameConstant):
            return bool(node.value)
        return False


def parse_python_file(filepath: str) -> list:
    """Парсва Python файл и връща модели"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
            
        tree = ast.parse(source)
        visitor = OdooASTVisitor()
        visitor.visit(tree)
        
        return visitor.models
    except SyntaxError:
        return []
    except Exception as e:
        return []


def analyze_odoo_source(paths: list) -> dict:
    """Анализира всички Python файлове"""
    all_models = {}
    files_processed = 0
    
    print("\n" + "=" * 60)
    print("АНАЛИЗ НА PYTHON КОД")
    print("=" * 60)
    
    for base_path, source_type in paths:
        path = Path(base_path)
        if not path.exists():
            continue
            
        print(f"\nАнализиране на {source_type}: {base_path}")
        
        py_files = list(path.rglob("*.py"))
        source_models = 0
        
        for py_file in py_files:
            if '__pycache__' in str(py_file):
                continue
            if 'test' in str(py_file).lower():
                continue
                
            models = parse_python_file(str(py_file))
            
            for model in models:
                # Определи модул
                rel_path = py_file.relative_to(path)
                parts = rel_path.parts
                if len(parts) > 0:
                    if parts[0] in ('addons', 'odoo'):
                        module = parts[1] if len(parts) > 1 else parts[0]
                    else:
                        module = parts[0]
                else:
                    module = 'base'
                    
                model.module = module
                
                # Определи бизнес домейн
                for key, domain in DOMAIN_MAP.items():
                    if key in model.name.lower() or key in module.lower():
                        model.business_domain = domain
                        break
                        
                # Добави/обедини
                if model.name not in all_models:
                    all_models[model.name] = model
                else:
                    # Merge fields
                    all_models[model.name].fields.update(model.fields)
                    if model.description and not all_models[model.name].description:
                        all_models[model.name].description = model.description
                        
                source_models += 1
                
            files_processed += 1
            
        print(f"  ✓ {source_models} модела от {len(py_files)} файла")
        
    print(f"\n✓ Общо модели: {len(all_models)}")
    print(f"✓ Обработени файлове: {files_processed}")
    
    return all_models


# ============================================
# TRAINING DATA GENERATOR
# ============================================

def get_system_prompt():
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
- Bill = Входяща фактура
- Warehouse = Склад
- Customer = Клиент
- Vendor = Доставчик
- Product = Продукт
- Quantity = Количество
- Unit Price = Единична цена
- Amount = Сума
- Total = Общо
- Tax = Данък
- Payment = Плащане
- Delivery = Доставка
- Shipment = Пратка
- Confirm = Потвърди
- Cancel = Отмени
- Validate = Валидирай
- Post = Осчетоводи
- Draft = Чернова
- Posted = Осчетоводено
- Paid = Платено
- Done = Приключено"""


def create_chatml_sample(source: str, target: str, 
                         model: str = '', field: str = '', 
                         domain: str = '', sample_type: str = '') -> dict:
    """Създава ChatML sample"""
    
    user_content = f'Преведи: "{source}"'
    
    context_parts = []
    if model:
        context_parts.append(f"Модел: {model}")
    if field:
        context_parts.append(f"Поле: {field}")
    if domain:
        context_parts.append(f"Домейн: {domain}")
        
    if context_parts:
        user_content += "\nКонтекст: " + ", ".join(context_parts)
    
    return {
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target}
        ]
    }


def generate_training_data(models: dict, translations: dict, output_dir: str):
    """Генерира training data"""
    
    print("\n" + "=" * 60)
    print("ГЕНЕРИРАНЕ НА TRAINING DATA")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    seen = set()
    stats = defaultdict(int)
    
    # 1. Samples от модели с контекст
    print("\nГенериране на samples от модели...")
    
    for model_name, model in models.items():
        for field_name, field_info in model.fields.items():
            # Field label
            if field_info.string:
                source = field_info.string
                target = translations.get(source)
                
                if target:
                    key = hashlib.md5(f"{source}|{target}|{model_name}|{field_name}".encode()).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        all_samples.append(create_chatml_sample(
                            source=source,
                            target=target,
                            model=model_name,
                            field=field_name,
                            domain=model.business_domain
                        ))
                        stats['field_label'] += 1
                        
            # Help text
            if field_info.help:
                source = field_info.help
                target = translations.get(source)
                
                if target:
                    key = hashlib.md5(f"{source}|{target}|help".encode()).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        all_samples.append(create_chatml_sample(
                            source=source,
                            target=target,
                            model=model_name,
                            domain=model.business_domain
                        ))
                        stats['help_text'] += 1
                        
            # Selection values
            if field_info.selection:
                for sel_key, sel_label in field_info.selection:
                    target = translations.get(sel_label)
                    if target:
                        key = hashlib.md5(f"{sel_label}|{target}|sel".encode()).hexdigest()
                        if key not in seen:
                            seen.add(key)
                            all_samples.append(create_chatml_sample(
                                source=sel_label,
                                target=target,
                                model=model_name,
                                field=field_name,
                                domain=model.business_domain
                            ))
                            stats['selection'] += 1
                            
    print(f"  ✓ {sum(stats.values())} samples с контекст")
    
    # 2. Останалите преводи без специфичен контекст
    print("\nДобавяне на останалите преводи...")
    
    used_sources = {s['messages'][1]['content'].split('"')[1] for s in all_samples if '"' in s['messages'][1]['content']}
    
    for source, target in translations.items():
        if source not in used_sources:
            key = hashlib.md5(f"{source}|{target}|general".encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                all_samples.append(create_chatml_sample(
                    source=source,
                    target=target
                ))
                stats['general'] += 1
                
    print(f"  ✓ {stats['general']} общи преводи")
    
    # 3. Augmentation - вариации
    print("\nAugmentation...")
    
    augmented = []
    for sample in all_samples[:5000]:  # Лимит за augmentation
        # Вариация без контекст
        if 'Контекст:' in sample['messages'][1]['content']:
            source_match = re.search(r'Преведи: "(.+?)"', sample['messages'][1]['content'])
            if source_match:
                source = source_match.group(1)
                target = sample['messages'][2]['content']
                key = hashlib.md5(f"{source}|{target}|nocontext".encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    augmented.append(create_chatml_sample(source, target))
                    stats['augmented'] += 1
                    
    all_samples.extend(augmented)
    print(f"  ✓ {stats['augmented']} augmented samples")
    
    # 4. Shuffle и split
    print("\nРазбъркване и разделяне...")
    
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # 5. Запис
    print("\nЗаписване на файлове...")
    
    with open(output_path / 'chatml_train.jsonl', 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
    with open(output_path / 'chatml_val.jsonl', 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
    # Alpaca формат
    alpaca_samples = []
    for sample in train_samples:
        user_msg = sample['messages'][1]['content']
        alpaca_samples.append({
            "instruction": "Преведи текста от Odoo ERP на български език.",
            "input": user_msg.replace('Преведи: ', '').replace('\nКонтекст:', '\n\nКонтекст:'),
            "output": sample['messages'][2]['content']
        })
        
    with open(output_path / 'alpaca_train.json', 'w', encoding='utf-8') as f:
        json.dump(alpaca_samples, f, ensure_ascii=False, indent=2)
        
    # Статистики
    stats_data = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'by_type': dict(stats)
    }
    
    with open(output_path / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"\nСтатистики:")
    print(f"  Общо samples: {len(all_samples)}")
    print(f"  Training: {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")
    print(f"\nПо тип:")
    for t, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    print(f"\nФайлове записани в: {output_dir}/")
    
    return stats_data


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("  ODOO TRANSLATION TRAINING DATA GENERATOR")
    print("=" * 60)
    print("\nИзточници:")
    for path, name in ODOO_PATHS:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {exists} {name}: {path}")
    
    # Провери пътища
    valid_paths = [(p, n) for p, n in ODOO_PATHS if Path(p).exists()]
    
    if not valid_paths:
        print("\n✗ Не са намерени валидни пътища!")
        print("Моля, редактирай ODOO_PATHS в скрипта.")
        sys.exit(1)
    
    # Събери преводи
    translations = collect_po_translations(valid_paths)
    
    # Анализирай код
    models = analyze_odoo_source(valid_paths)
    
    # Генерирай training data
    stats = generate_training_data(models, translations, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("СЛЕДВАЩИ СТЪПКИ")
    print("=" * 60)
    print(f"""
1. Архивирай training data:
   zip -r training_data.zip {OUTPUT_DIR}/

2. Качи в RunPod/Vast.ai

3. В cloud машината:
   unzip training_data.zip
   python cloud_training.py

4. Свали GGUF файла и използвай с Ollama:
   ollama create odoo-translate-bg -f Modelfile
""")


if __name__ == '__main__':
    main()
