#!/usr/bin/env python3
"""
Batch Code Analyzer
===================
По-ефективен анализ - групира код и анализира на batch-ове.
Вместо 1 LLM call на файл, групира по модул.

Стъпки:
1. Извлича всички модели/полета/методи (бързо, без LLM)
2. Групира по модул
3. Изпраща обобщен код към LLM за анализ
4. Генерира богат контекст за training
"""

import os
import sys
import json
import ast
import re
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import defaultdict
import urllib.request
import urllib.error

# ============================================
# КОНФИГУРАЦИЯ  
# ============================================

ODOO_PATHS = [
    ("/opt/odoo/base/odoo-18.0", "community"),
    ("/opt/odoo/ee/odoo-18.0", "enterprise"),
    ("/opt/odoo/oca/odoo-18.0", "oca"),
]

LLM_MODEL = "qwen2.5:14b"  # или qwen2.5:7b за по-бързо
LLM_URL = "http://localhost:11434"
OUTPUT_DIR = "./training_data"

FIELD_TYPES = {
    'Char', 'Text', 'Html', 'Integer', 'Float', 'Monetary',
    'Boolean', 'Date', 'Datetime', 'Binary', 'Image',
    'Selection', 'Many2one', 'One2many', 'Many2many',
    'Reference', 'Properties', 'Json'
}


# ============================================
# LLM CLIENT
# ============================================

def llm_analyze(prompt: str, system: str = None, timeout: int = 120) -> str:
    """Изпраща prompt към Ollama"""
    
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 3000}
    }
    if system:
        payload["system"] = system
        
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{LLM_URL}/api/generate",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read())
            return result.get('response', '')
    except Exception as e:
        print(f"    LLM грешка: {e}")
        return ""


def check_ollama():
    """Проверява Ollama (работи и с Docker)"""
    try:
        req = urllib.request.Request(f"{LLM_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            models = [m['name'] for m in data.get('models', [])]
            print(f"✓ Ollama API достъпен на {LLM_URL}")
            print(f"  Налични модели: {models}")
            
            # Провери дали моделът е наличен
            model_base = LLM_MODEL.split(':')[0]
            if any(model_base in m for m in models):
                print(f"✓ Модел {LLM_MODEL} е готов")
                return True
            else:
                print(f"⚠ Модел {LLM_MODEL} не е намерен")
                print(f"  Изпълни: docker exec -it ollama-amdgpu ollama pull {LLM_MODEL}")
                return False
    except urllib.error.URLError as e:
        print(f"✗ Ollama API не е достъпен на {LLM_URL}")
        print(f"  Грешка: {e}")
        print("  Провери дали Docker контейнерът работи:")
        print("    docker ps | grep ollama")
        return False
    except Exception as e:
        print(f"✗ Грешка при връзка с Ollama: {e}")
        return False


# ============================================
# CODE EXTRACTION (Fast, no LLM)
# ============================================

def extract_module_info(module_path: Path) -> dict:
    """Извлича информация от цял модул (бързо)"""
    
    module_info = {
        'name': module_path.name,
        'path': str(module_path),
        'models': {},
        'all_strings': set(),
        'manifest': None,
    }
    
    # Прочети manifest
    manifest_path = module_path / '__manifest__.py'
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Извлечи dict
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                module_info['manifest'] = eval(match.group())
        except:
            pass
            
    # Анализирай Python файлове
    for py_file in module_path.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
                
            tree = ast.parse(source)
            
            # Извлечи класове
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    model_info = extract_class_info(node, source)
                    if model_info and model_info.get('_name'):
                        model_name = model_info['_name']
                        if model_name not in module_info['models']:
                            module_info['models'][model_name] = model_info
                        else:
                            # Merge fields
                            module_info['models'][model_name]['fields'].update(
                                model_info.get('fields', {})
                            )
                            
            # Извлечи translatable strings
            for match in re.finditer(r'_\(["\'](.+?)["\']\)', source):
                module_info['all_strings'].add(match.group(1))
                
        except:
            continue
            
    module_info['all_strings'] = list(module_info['all_strings'])
    return module_info


def extract_class_info(node: ast.ClassDef, source: str) -> dict:
    """Извлича информация от клас"""
    
    info = {
        'class_name': node.name,
        '_name': None,
        '_inherit': [],
        '_description': None,
        'fields': {},
        'methods': [],
    }
    
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    val = get_value(item.value)
                    if target.id == '_name':
                        info['_name'] = val
                    elif target.id == '_inherit':
                        info['_inherit'] = val if isinstance(val, list) else [val] if val else []
                    elif target.id == '_description':
                        info['_description'] = val
                    else:
                        # Поле
                        field_info = extract_field_info(target.id, item.value)
                        if field_info:
                            info['fields'][target.id] = field_info
                            
        elif isinstance(item, ast.FunctionDef):
            method_info = {
                'name': item.name,
                'decorators': [get_decorator_name(d) for d in item.decorator_list],
                'docstring': ast.get_docstring(item),
                'errors': extract_error_messages(item, source),
            }
            info['methods'].append(method_info)
            
    if not info['_name'] and info['_inherit']:
        info['_name'] = info['_inherit'][0]
        
    return info if info['_name'] else None


def extract_field_info(name: str, node) -> dict:
    """Извлича информация за поле"""
    
    if not isinstance(node, ast.Call):
        return None
        
    field_type = None
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in FIELD_TYPES:
            field_type = node.func.attr
    elif isinstance(node.func, ast.Name):
        if node.func.id in FIELD_TYPES:
            field_type = node.func.id
            
    if not field_type:
        return None
        
    info = {
        'name': name,
        'type': field_type,
        'string': None,
        'help': None,
        'selection': None,
        'comodel_name': None,
        'compute': None,
        'related': None,
    }
    
    for kw in node.keywords:
        val = get_value(kw.value)
        if kw.arg == 'string':
            info['string'] = val
        elif kw.arg == 'help':
            info['help'] = val
        elif kw.arg == 'comodel_name':
            info['comodel_name'] = val
        elif kw.arg == 'compute':
            info['compute'] = val
        elif kw.arg == 'related':
            info['related'] = val
        elif kw.arg == 'selection':
            info['selection'] = extract_selection(kw.value)
            
    return info


def extract_selection(node) -> list:
    """Извлича selection стойности"""
    selections = []
    if isinstance(node, ast.List):
        for item in node.elts:
            if isinstance(item, ast.Tuple) and len(item.elts) >= 2:
                key = get_value(item.elts[0])
                label = get_value(item.elts[1])
                if key and label:
                    selections.append({'key': key, 'label': label})
    return selections


def extract_error_messages(node: ast.FunctionDef, source: str) -> list:
    """Извлича error messages от метод"""
    errors = []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
            exc_name = get_name(child.exc.func)
            if exc_name in ('ValidationError', 'UserError', 'AccessError'):
                if child.exc.args:
                    msg = get_value(child.exc.args[0])
                    if msg and isinstance(msg, str):
                        errors.append({'type': exc_name, 'message': msg})
    return errors


def get_value(node) -> Any:
    """Извлича стойност"""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.List):
        return [get_value(el) for el in node.elts]
    if isinstance(node, ast.Name):
        return node.id
    return None


def get_name(node) -> str:
    """Извлича име"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def get_decorator_name(node) -> str:
    """Извлича декоратор"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return get_decorator_name(node.func)
    return ""


# ============================================
# LLM BATCH ANALYSIS
# ============================================

def analyze_module_with_llm(module_info: dict) -> dict:
    """Анализира модул с LLM"""
    
    # Подготви обобщение на модула
    summary_lines = []
    
    # Manifest info
    if module_info.get('manifest'):
        m = module_info['manifest']
        summary_lines.append(f"Модул: {module_info['name']}")
        if m.get('name'):
            summary_lines.append(f"Име: {m['name']}")
        if m.get('summary'):
            summary_lines.append(f"Описание: {m['summary']}")
        if m.get('category'):
            summary_lines.append(f"Категория: {m['category']}")
        if m.get('depends'):
            summary_lines.append(f"Зависимости: {m['depends'][:5]}")
            
    summary_lines.append("\nМодели:")
    
    # Модели и полета
    for model_name, model in list(module_info['models'].items())[:10]:
        summary_lines.append(f"\n  {model_name}:")
        if model.get('_description'):
            summary_lines.append(f"    Описание: {model['_description']}")
            
        # Полета с labels
        fields_with_strings = [
            f for f in model.get('fields', {}).values() 
            if f.get('string')
        ]
        if fields_with_strings:
            summary_lines.append("    Полета:")
            for f in fields_with_strings[:15]:
                line = f"      - {f['name']} ({f['type']}): \"{f['string']}\""
                if f.get('help'):
                    line += f" [{f['help'][:50]}...]"
                summary_lines.append(line)
                
        # Selection полета
        selection_fields = [
            f for f in model.get('fields', {}).values()
            if f.get('selection')
        ]
        if selection_fields:
            for f in selection_fields[:5]:
                vals = ", ".join(s['label'] for s in f['selection'][:5])
                summary_lines.append(f"      - {f['name']} selection: {vals}")
                
    module_summary = "\n".join(summary_lines)
    
    # LLM prompt
    system = """Ти си експерт по Odoo ERP. Анализирай модула и върни JSON:
{
  "business_domain": "Бизнес област (Продажби/Покупки/Счетоводство/Склад/HR/Производство/CRM/Проекти/...)",
  "business_processes": ["списък с бизнес процеси в които участва"],
  "typical_user": "Типичен потребител (търговец/счетоводител/склададжия/HR специалист/...)",
  "module_purpose": "Кратко описание какво прави модула",
  "models_analysis": {
    "model.name": {
      "purpose": "Какво представлява модела",
      "workflow": "Описание на workflow ако има",
      "fields_context": {
        "field_name": {
          "meaning": "Какво означава за потребителя",
          "translation_hint": "Подсказка за български превод"
        }
      }
    }
  }
}
Върни САМО валиден JSON."""

    prompt = f"""Анализирай този Odoo модул:

{module_summary}

Върни JSON анализ:"""

    response = llm_analyze(prompt, system, timeout=180)
    
    # Парсни JSON
    try:
        response = re.sub(r'^```json\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response)
        return json.loads(response)
    except:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {}


# ============================================
# PO TRANSLATIONS
# ============================================

def load_translations(paths: list) -> dict:
    """Зарежда PO преводи"""
    translations = {}
    
    for base_path, _ in paths:
        path = Path(base_path)
        if not path.exists():
            continue
            
        for po_file in path.rglob("bg*.po"):
            try:
                with open(po_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                pattern = r'msgid\s+"((?:[^"\\]|\\.)*)"\s*\nmsgstr\s+"((?:[^"\\]|\\.)*)"'
                for match in re.finditer(pattern, content):
                    source = match.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
                    target = match.group(2).replace('\\n', '\n').replace('\\"', '"').strip()
                    if source and target and source != target:
                        translations[source] = target
            except:
                continue
                
    return translations


# ============================================
# TRAINING DATA GENERATION
# ============================================

def generate_training_samples(modules_data: list, translations: dict) -> list:
    """Генерира training samples"""
    
    samples = []
    seen = set()
    
    for module in modules_data:
        module_name = module['info']['name']
        analysis = module.get('analysis', {})
        
        business_domain = analysis.get('business_domain', '')
        module_purpose = analysis.get('module_purpose', '')
        typical_user = analysis.get('typical_user', '')
        models_analysis = analysis.get('models_analysis', {})
        
        for model_name, model in module['info'].get('models', {}).items():
            model_analysis = models_analysis.get(model_name, {})
            model_purpose = model_analysis.get('purpose', '')
            fields_context = model_analysis.get('fields_context', {})
            
            for field_name, field in model.get('fields', {}).items():
                if not field.get('string'):
                    continue
                    
                source = field['string']
                target = translations.get(source)
                if not target:
                    continue
                    
                key = hashlib.md5(f"{source}|{target}|{model_name}".encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                
                # Контекст от LLM анализ
                field_context = fields_context.get(field_name, {})
                field_meaning = field_context.get('meaning', '')
                translation_hint = field_context.get('translation_hint', '')
                
                sample = create_sample(
                    source=source,
                    target=target,
                    model=model_name,
                    field=field_name,
                    field_type=field['type'],
                    domain=business_domain,
                    process=module_purpose,
                    meaning=field_meaning,
                    hint=translation_hint,
                    sample_type='field_label'
                )
                samples.append(sample)
                
                # Help text
                if field.get('help'):
                    help_target = translations.get(field['help'])
                    if help_target:
                        key2 = hashlib.md5(f"{field['help']}|{help_target}".encode()).hexdigest()
                        if key2 not in seen:
                            seen.add(key2)
                            samples.append(create_sample(
                                source=field['help'],
                                target=help_target,
                                model=model_name,
                                field=field_name,
                                domain=business_domain,
                                meaning=f"Помощен текст за {source}",
                                sample_type='help_text'
                            ))
                            
                # Selection values
                if field.get('selection'):
                    for sel in field['selection']:
                        sel_label = sel.get('label')
                        if sel_label:
                            sel_target = translations.get(sel_label)
                            if sel_target:
                                key3 = hashlib.md5(f"{sel_label}|{sel_target}".encode()).hexdigest()
                                if key3 not in seen:
                                    seen.add(key3)
                                    samples.append(create_sample(
                                        source=sel_label,
                                        target=sel_target,
                                        model=model_name,
                                        field=field_name,
                                        domain=business_domain,
                                        meaning=f"Стойност за {source}",
                                        sample_type='selection_value'
                                    ))
                                    
            # Error messages
            for method in model.get('methods', []):
                for error in method.get('errors', []):
                    msg = error.get('message')
                    if msg:
                        msg_target = translations.get(msg)
                        if msg_target:
                            key4 = hashlib.md5(f"{msg}|{msg_target}".encode()).hexdigest()
                            if key4 not in seen:
                                seen.add(key4)
                                samples.append(create_sample(
                                    source=msg,
                                    target=msg_target,
                                    model=model_name,
                                    domain=business_domain,
                                    meaning=f"Съобщение от {method['name']}",
                                    sample_type='error_message'
                                ))
                                
    return samples


def create_sample(source: str, target: str, model: str = '', field: str = '',
                  field_type: str = '', domain: str = '', process: str = '',
                  meaning: str = '', hint: str = '', sample_type: str = '') -> dict:
    """Създава training sample с богат контекст"""
    
    system = """Ти си експертен преводач за Odoo ERP система. Превеждаш от английски на български.

Правила:
1. Използвай установената Odoo терминология
2. Съобразявай се с бизнес контекста
3. Преводът да е естествен и професионален
4. Отговаряй САМО с превода"""

    user_parts = [f'Преведи: "{source}"']
    
    context_lines = []
    if model:
        context_lines.append(f"Модел: {model}")
    if domain:
        context_lines.append(f"Домейн: {domain}")
    if process:
        context_lines.append(f"Процес: {process}")
    if field:
        context_lines.append(f"Поле: {field}")
    if field_type:
        context_lines.append(f"Тип поле: {field_type}")
    if meaning:
        context_lines.append(f"Значение: {meaning}")
    if hint:
        context_lines.append(f"Подсказка: {hint}")
    if sample_type:
        type_names = {
            'field_label': 'Етикет на поле',
            'help_text': 'Помощен текст', 
            'selection_value': 'Стойност от списък',
            'error_message': 'Съобщение за грешка',
        }
        context_lines.append(f"Тип: {type_names.get(sample_type, sample_type)}")
        
    if context_lines:
        user_parts.append("\nКонтекст:")
        user_parts.extend(f"  {line}" for line in context_lines)
        
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_parts)},
            {"role": "assistant", "content": target}
        ]
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("  BATCH SEMANTIC ODOO ANALYZER")
    print("=" * 60)
    
    # Провери Ollama
    if not check_ollama():
        sys.exit(1)
        
    # Провери пътища
    print("\nПроверка на Odoo директории...")
    valid_paths = []
    for path, name in ODOO_PATHS:
        if Path(path).exists():
            print(f"  ✓ {name}: {path}")
            valid_paths.append((path, name))
        else:
            print(f"  ✗ {name}: {path}")
            
    if not valid_paths:
        print("\n✗ Не са намерени Odoo директории!")
        sys.exit(1)
        
    # Зареди преводи
    print("\n" + "=" * 60)
    print("ЗАРЕЖДАНЕ НА ПРЕВОДИ")
    print("=" * 60)
    translations = load_translations(valid_paths)
    print(f"✓ Заредени преводи: {len(translations)}")
    
    # Извлечи модули (бързо, без LLM)
    print("\n" + "=" * 60)
    print("ИЗВЛИЧАНЕ НА МОДУЛИ (бързо)")
    print("=" * 60)
    
    all_modules = []
    
    for base_path, source_type in valid_paths:
        path = Path(base_path)
        
        # Намери модули (директории с __manifest__.py)
        if (path / 'addons').exists():
            modules_dir = path / 'addons'
        elif (path / 'odoo' / 'addons').exists():
            modules_dir = path / 'odoo' / 'addons'
        else:
            modules_dir = path
            
        module_dirs = [
            d for d in modules_dir.iterdir() 
            if d.is_dir() and (d / '__manifest__.py').exists()
        ]
        
        print(f"\n{source_type}: {len(module_dirs)} модула")
        
        for module_dir in module_dirs:
            module_info = extract_module_info(module_dir)
            if module_info['models']:
                all_modules.append({
                    'info': module_info,
                    'source': source_type,
                    'analysis': {}
                })
                
    print(f"\n✓ Общо модули с модели: {len(all_modules)}")
    total_models = sum(len(m['info']['models']) for m in all_modules)
    print(f"✓ Общо Odoo модели: {total_models}")
    
    # LLM анализ на модули
    print("\n" + "=" * 60)
    print("LLM АНАЛИЗ НА МОДУЛИ")
    print("=" * 60)
    print(f"Това ще отнеме ~{len(all_modules) * 0.5:.0f}-{len(all_modules) * 1:.0f} минути")
    
    for i, module in enumerate(all_modules):
        module_name = module['info']['name']
        models_count = len(module['info']['models'])
        
        print(f"\n[{i+1}/{len(all_modules)}] {module_name} ({models_count} модела)...", end=" ", flush=True)
        
        analysis = analyze_module_with_llm(module['info'])
        module['analysis'] = analysis
        
        if analysis:
            domain = analysis.get('business_domain', 'N/A')
            print(f"✓ ({domain})")
        else:
            print("⚠ (без анализ)")
            
        # Малка пауза
        time.sleep(0.5)
        
    # Генерирай training data
    print("\n" + "=" * 60)
    print("ГЕНЕРИРАНЕ НА TRAINING DATA")
    print("=" * 60)
    
    samples = generate_training_samples(all_modules, translations)
    print(f"✓ Генерирани samples: {len(samples)}")
    
    # Запиши
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    import random
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * 0.9)
    train = samples[:split_idx]
    val = samples[split_idx:]
    
    with open(output_path / 'chatml_train.jsonl', 'w', encoding='utf-8') as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
            
    with open(output_path / 'chatml_val.jsonl', 'w', encoding='utf-8') as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
            
    # Запази анализа
    analysis_data = {
        'modules': [{
            'name': m['info']['name'],
            'source': m['source'],
            'models_count': len(m['info']['models']),
            'analysis': m['analysis']
        } for m in all_modules]
    }
    
    with open(output_path / 'modules_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n✓ Train: {len(train)}")
    print(f"✓ Val: {len(val)}")
    print(f"✓ Записано в: {OUTPUT_DIR}/")
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"""
Следващи стъпки:
1. zip -r training_data.zip {OUTPUT_DIR}/
2. Качи в RunPod
3. Стартирай обучение
""")


if __name__ == '__main__':
    main()
