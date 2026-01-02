#!/usr/bin/env python3
"""
Deep Semantic Code Analyzer
============================
Анализира Odoo source код с помощта на локален LLM модел.
За всеки файл/клас/метод/поле генерира семантичен контекст:
- Какво прави кодът
- В какъв бизнес процес участва
- Как се използва
- Контекст за превод

Изисква: Ollama с модел (qwen2.5, deepseek-coder, codellama)
"""

import os
import sys
import json
import ast
import re
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

CONFIG = {
    # Пътища към Odoo
    "odoo_paths": [
        ("/opt/odoo/base/odoo-18.0", "community"),
        ("/opt/odoo/ee/odoo-18.0", "enterprise"),
        ("/opt/odoo/oca/odoo-18.0", "oca"),
    ],
    
    # LLM за анализ на код
    "llm_model": "qwen2.5:14b",  # или deepseek-coder:6.7b, codellama:13b
    "llm_base_url": "http://localhost:11434",
    
    # Output
    "output_dir": "./deep_analysis",
    "training_output": "./training_data",
    
    # Лимити
    "max_file_size": 100000,  # bytes
    "max_code_chunk": 4000,   # chars за LLM
    "batch_size": 10,         # файла наведнъж
    "llm_timeout": 60,        # секунди
    
    # Паралелизъм
    "max_workers": 4,
}


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class FieldAnalysis:
    """Дълбок анализ на поле"""
    name: str
    field_type: str
    string: Optional[str] = None
    help: Optional[str] = None
    
    # Семантичен анализ от LLM
    purpose: str = ""           # Какво е предназначението
    business_context: str = ""  # В какъв бизнес процес участва
    data_meaning: str = ""      # Какво означават данните
    usage_scenario: str = ""    # Как се използва от потребителя
    translation_hints: str = "" # Подсказки за превод
    
    # Технически детайли
    comodel_name: Optional[str] = None
    selection: Optional[list] = None
    compute_method: Optional[str] = None
    related_fields: List[str] = field(default_factory=list)


@dataclass
class MethodAnalysis:
    """Дълбок анализ на метод"""
    name: str
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    
    # Семантичен анализ
    purpose: str = ""
    business_action: str = ""     # Какво бизнес действие извършва
    triggers: str = ""            # Кога се извиква
    side_effects: str = ""        # Какви промени прави
    user_messages: List[str] = field(default_factory=list)  # Съобщения към потребителя


@dataclass
class ModelAnalysis:
    """Дълбок анализ на модел"""
    name: str
    technical_name: str
    description: Optional[str] = None
    inherit: List[str] = field(default_factory=list)
    
    # Семантичен анализ
    purpose: str = ""
    business_domain: str = ""
    business_process: str = ""
    user_role: str = ""           # Кой потребител работи с този модел
    workflow_description: str = ""
    
    # Съдържание
    fields: Dict[str, FieldAnalysis] = field(default_factory=dict)
    methods: Dict[str, MethodAnalysis] = field(default_factory=dict)
    
    # Метаданни
    module: str = ""
    file_path: str = ""


@dataclass
class TranslationContext:
    """Контекст за превод на конкретен string"""
    source_string: str
    string_type: str  # field_label, help_text, error_message, button, menu, etc.
    
    # Пълен контекст
    model_name: str
    model_purpose: str
    field_name: Optional[str]
    field_purpose: str
    method_name: Optional[str]
    
    business_domain: str
    business_process: str
    user_action: str
    
    # Семантика
    semantic_meaning: str
    translation_hints: str
    similar_translations: List[tuple] = field(default_factory=list)
    
    # Превод ако съществува
    existing_translation: Optional[str] = None


# ============================================
# LLM CLIENT
# ============================================

class OllamaClient:
    """Клиент за Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:14b"):
        self.base_url = base_url
        self.model = model
        self._check_connection()
        
    def _check_connection(self):
        """Проверява връзката с Ollama"""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read())
                models = [m['name'] for m in data.get('models', [])]
                
                if not any(self.model in m for m in models):
                    print(f"⚠ Модел {self.model} не е намерен. Налични: {models[:5]}")
                    print(f"  Изпълни: ollama pull {self.model}")
                else:
                    print(f"✓ Ollama свързан, модел: {self.model}")
        except Exception as e:
            print(f"✗ Грешка при връзка с Ollama: {e}")
            print("  Стартирай Ollama: ollama serve")
            sys.exit(1)
            
    def generate(self, prompt: str, system: str = None, temperature: float = 0.3) -> str:
        """Генерира отговор"""
        import urllib.request
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2000,
            }
        }
        
        if system:
            payload["system"] = system
            
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=CONFIG["llm_timeout"]) as response:
                result = json.loads(response.read())
                return result.get('response', '')
        except Exception as e:
            print(f"  ⚠ LLM грешка: {e}")
            return ""
            
    def analyze_code(self, code: str, context: str = "") -> dict:
        """Анализира код и връща структуриран резултат"""
        
        system_prompt = """Ти си експерт по Odoo ERP разработка. Анализирай Python кода и върни JSON с:
{
  "module_purpose": "Кратко описание какво прави модулът",
  "business_domain": "Бизнес област (Продажби/Покупки/Счетоводство/Склад/HR/и т.н.)",
  "business_process": "Бизнес процес в който участва",
  "user_role": "Типичен потребител (търговец/счетоводител/склададжия/и т.н.)",
  "models": [
    {
      "name": "technical.model.name",
      "purpose": "Какво представлява този модел",
      "workflow": "Описание на workflow ако има"
    }
  ],
  "fields": [
    {
      "name": "field_name",
      "label": "Field Label ако има",
      "purpose": "Какво съхранява и защо",
      "user_meaning": "Какво означава за потребителя",
      "translation_hint": "Подсказка за превод на български"
    }
  ],
  "methods": [
    {
      "name": "method_name", 
      "action": "Какво действие извършва",
      "trigger": "Кога се извиква"
    }
  ],
  "user_messages": ["списък с текстове които потребителя вижда"]
}

Отговори САМО с валиден JSON, без markdown formatting."""

        prompt = f"""Анализирай този Odoo код:

{context}

```python
{code[:CONFIG["max_code_chunk"]]}
```

Върни JSON анализ:"""

        response = self.generate(prompt, system_prompt)
        
        # Парсни JSON
        try:
            # Премахни markdown ако има
            response = re.sub(r'^```json\s*', '', response)
            response = re.sub(r'\s*```$', '', response)
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError:
            # Опитай да извлечеш JSON
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return {}


# ============================================
# CODE EXTRACTOR
# ============================================

class OdooCodeExtractor:
    """Извлича и структурира Odoo код за анализ"""
    
    FIELD_TYPES = {
        'Char', 'Text', 'Html', 'Integer', 'Float', 'Monetary',
        'Boolean', 'Date', 'Datetime', 'Binary', 'Image',
        'Selection', 'Many2one', 'One2many', 'Many2many',
        'Reference', 'Properties', 'PropertiesDefinition', 'Json'
    }
    
    def __init__(self):
        self.files_processed = 0
        self.errors = []
        
    def extract_from_file(self, filepath: str) -> dict:
        """Извлича структурирана информация от Python файл"""
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
                
            if len(source) > CONFIG["max_file_size"]:
                return {'error': 'file_too_large'}
                
            tree = ast.parse(source)
            
            result = {
                'filepath': filepath,
                'source_code': source,
                'imports': self._extract_imports(tree),
                'classes': [],
                'functions': [],
                'strings': self._extract_strings(source),
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class(node, source)
                    if class_info:
                        result['classes'].append(class_info)
                elif isinstance(node, ast.FunctionDef) and not hasattr(node, '_in_class'):
                    func_info = self._extract_function(node)
                    result['functions'].append(func_info)
                    
            self.files_processed += 1
            return result
            
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {filepath}: {e}")
            return {'error': 'syntax_error'}
        except Exception as e:
            self.errors.append(f"Error in {filepath}: {e}")
            return {'error': str(e)}
            
    def _extract_imports(self, tree) -> list:
        """Извлича imports за контекст"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
        
    def _extract_class(self, node: ast.ClassDef, source: str) -> dict:
        """Извлича информация за клас"""
        
        class_info = {
            'name': node.name,
            'bases': [self._get_name(b) for b in node.bases],
            'docstring': ast.get_docstring(node),
            'odoo_model': None,
            'odoo_inherit': [],
            'odoo_description': None,
            'fields': [],
            'methods': [],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
        }
        
        for item in node.body:
            # Odoo атрибути
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id == '_name':
                            class_info['odoo_model'] = self._get_value(item.value)
                        elif target.id == '_inherit':
                            val = self._get_value(item.value)
                            class_info['odoo_inherit'] = val if isinstance(val, list) else [val] if val else []
                        elif target.id == '_description':
                            class_info['odoo_description'] = self._get_value(item.value)
                        else:
                            # Поле
                            field_info = self._extract_field(target.id, item.value)
                            if field_info:
                                class_info['fields'].append(field_info)
                                
            # Методи
            elif isinstance(item, ast.FunctionDef):
                method_info = self._extract_method(item, source)
                class_info['methods'].append(method_info)
                
        # Филтрирай не-Odoo класове
        if not class_info['odoo_model'] and not class_info['odoo_inherit']:
            if not any('Model' in b for b in class_info['bases']):
                return None
                
        return class_info
        
    def _extract_field(self, name: str, node) -> dict:
        """Извлича информация за поле"""
        
        if not isinstance(node, ast.Call):
            return None
            
        field_type = None
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FIELD_TYPES:
                field_type = node.func.attr
        elif isinstance(node.func, ast.Name):
            if node.func.id in self.FIELD_TYPES:
                field_type = node.func.id
                
        if not field_type:
            return None
            
        field_info = {
            'name': name,
            'type': field_type,
            'string': None,
            'help': None,
            'selection': None,
            'comodel_name': None,
            'compute': None,
            'related': None,
            'required': False,
            'readonly': False,
            'default': None,
        }
        
        for kw in node.keywords:
            if kw.arg == 'string':
                field_info['string'] = self._get_value(kw.value)
            elif kw.arg == 'help':
                field_info['help'] = self._get_value(kw.value)
            elif kw.arg == 'selection':
                field_info['selection'] = self._extract_selection(kw.value)
            elif kw.arg == 'comodel_name':
                field_info['comodel_name'] = self._get_value(kw.value)
            elif kw.arg == 'compute':
                field_info['compute'] = self._get_value(kw.value)
            elif kw.arg == 'related':
                field_info['related'] = self._get_value(kw.value)
            elif kw.arg == 'required':
                field_info['required'] = self._get_value(kw.value) is True
            elif kw.arg == 'readonly':
                field_info['readonly'] = self._get_value(kw.value) is True
            elif kw.arg == 'default':
                field_info['default'] = str(self._get_value(kw.value))
                
        return field_info
        
    def _extract_method(self, node: ast.FunctionDef, source: str) -> dict:
        """Извлича информация за метод"""
        
        # Извлечи source code на метода
        try:
            method_source = ast.get_source_segment(source, node)
        except:
            method_source = None
            
        method_info = {
            'name': node.name,
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'args': [arg.arg for arg in node.args.args],
            'source': method_source,
            'raises_errors': [],
            'returns': None,
            'calls': [],
            'field_access': [],
        }
        
        # Анализирай тялото
        for child in ast.walk(node):
            # ValidationError, UserError
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Call):
                    exc_name = self._get_name(child.exc.func)
                    if exc_name in ('ValidationError', 'UserError', 'AccessError'):
                        # Извлечи съобщението
                        if child.exc.args:
                            msg = self._get_value(child.exc.args[0])
                            if msg:
                                method_info['raises_errors'].append({
                                    'type': exc_name,
                                    'message': msg
                                })
                                
            # Извиквания на методи
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    method_info['calls'].append(child.func.attr)
                    
            # Достъп до self.field
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name) and child.value.id == 'self':
                    method_info['field_access'].append(child.attr)
                    
        return method_info
        
    def _extract_selection(self, node) -> list:
        """Извлича selection стойности"""
        selections = []
        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) >= 2:
                    key = self._get_value(item.elts[0])
                    label = self._get_value(item.elts[1])
                    if key and label:
                        selections.append({'key': key, 'label': label})
        return selections
        
    def _extract_strings(self, source: str) -> list:
        """Извлича всички translatable strings"""
        strings = []
        
        # _("string") или _('string')
        pattern = r'_\(\s*["\'](.+?)["\']\s*\)'
        for match in re.finditer(pattern, source):
            strings.append({
                'text': match.group(1),
                'type': 'translatable'
            })
            
        return strings
        
    def _extract_function(self, node: ast.FunctionDef) -> dict:
        """Извлича standalone функция"""
        return {
            'name': node.name,
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'args': [arg.arg for arg in node.args.args],
        }
        
    def _get_value(self, node) -> Any:
        """Извлича стойност от AST node"""
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return [self._get_value(el) for el in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._get_value(el) for el in node.elts)
        return None
        
    def _get_name(self, node) -> str:
        """Извлича име от AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""
        
    def _get_decorator_name(self, node) -> str:
        """Извлича име на декоратор"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""


# ============================================
# DEEP ANALYZER
# ============================================

class DeepSemanticAnalyzer:
    """Дълбок семантичен анализ с LLM"""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.extractor = OdooCodeExtractor()
        self.analyzed_files = {}
        self.all_models = {}
        self.all_translations = {}
        
    def analyze_directory(self, path: str, source_type: str = ""):
        """Анализира директория с Odoo модули"""
        
        path = Path(path)
        if not path.exists():
            print(f"⚠ Път не съществува: {path}")
            return
            
        print(f"\n{'=' * 60}")
        print(f"ДЪЛБОК АНАЛИЗ: {path}")
        print(f"{'=' * 60}")
        
        # Намери всички Python файлове
        py_files = []
        for py_file in path.rglob("*.py"):
            if '__pycache__' not in str(py_file):
                if 'test' not in str(py_file).lower():
                    py_files.append(py_file)
                    
        print(f"Намерени Python файлове: {len(py_files)}")
        
        # Анализирай на batch-ове
        batch_size = CONFIG["batch_size"]
        total_batches = (len(py_files) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(py_files))
            batch = py_files[start:end]
            
            print(f"\nBatch {batch_idx + 1}/{total_batches} ({len(batch)} файла)...")
            
            for py_file in batch:
                self._analyze_file(py_file, source_type)
                
            # Progress
            done = min(end, len(py_files))
            print(f"  Прогрес: {done}/{len(py_files)} файла")
            
    def _analyze_file(self, filepath: Path, source_type: str):
        """Анализира един файл с LLM"""
        
        # Извлечи структура
        extracted = self.extractor.extract_from_file(str(filepath))
        
        if 'error' in extracted:
            return
            
        # Определи модул
        parts = filepath.parts
        module = ""
        for i, part in enumerate(parts):
            if part in ('addons', 'odoo') and i + 1 < len(parts):
                module = parts[i + 1]
                break
        if not module:
            module = parts[-2] if len(parts) > 1 else parts[-1]
            
        # Ако има Odoo класове, анализирай с LLM
        if extracted.get('classes'):
            for class_info in extracted['classes']:
                if class_info.get('odoo_model') or class_info.get('odoo_inherit'):
                    self._analyze_class_with_llm(class_info, extracted, module, source_type, filepath)
                    
    def _analyze_class_with_llm(self, class_info: dict, file_info: dict, 
                                module: str, source_type: str, filepath: Path):
        """Анализира клас с LLM"""
        
        model_name = class_info.get('odoo_model') or (
            class_info['odoo_inherit'][0] if class_info.get('odoo_inherit') else class_info['name']
        )
        
        # Подготви код за анализ
        code_summary = self._prepare_code_for_llm(class_info)
        
        # Контекст
        context = f"Модул: {module}\nФайл: {filepath.name}\nТип: {source_type}"
        
        # Извикай LLM
        print(f"    Анализ: {model_name}...", end=" ", flush=True)
        
        try:
            llm_analysis = self.llm.analyze_code(code_summary, context)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
            llm_analysis = {}
            
        # Създай ModelAnalysis
        model_analysis = ModelAnalysis(
            name=class_info.get('odoo_description') or model_name,
            technical_name=model_name,
            description=class_info.get('docstring'),
            inherit=class_info.get('odoo_inherit', []),
            purpose=llm_analysis.get('module_purpose', ''),
            business_domain=llm_analysis.get('business_domain', ''),
            business_process=llm_analysis.get('business_process', ''),
            user_role=llm_analysis.get('user_role', ''),
            module=module,
            file_path=str(filepath),
        )
        
        # Анализирай полета
        llm_fields = {f.get('name'): f for f in llm_analysis.get('fields', [])}
        
        for field_info in class_info.get('fields', []):
            fname = field_info['name']
            llm_field = llm_fields.get(fname, {})
            
            field_analysis = FieldAnalysis(
                name=fname,
                field_type=field_info['type'],
                string=field_info.get('string'),
                help=field_info.get('help'),
                purpose=llm_field.get('purpose', ''),
                business_context=llm_analysis.get('business_process', ''),
                data_meaning=llm_field.get('user_meaning', ''),
                translation_hints=llm_field.get('translation_hint', ''),
                comodel_name=field_info.get('comodel_name'),
                selection=field_info.get('selection'),
                compute_method=field_info.get('compute'),
            )
            
            model_analysis.fields[fname] = field_analysis
            
        # Анализирай методи
        llm_methods = {m.get('name'): m for m in llm_analysis.get('methods', [])}
        
        for method_info in class_info.get('methods', []):
            mname = method_info['name']
            llm_method = llm_methods.get(mname, {})
            
            method_analysis = MethodAnalysis(
                name=mname,
                decorators=method_info.get('decorators', []),
                docstring=method_info.get('docstring'),
                purpose=llm_method.get('action', ''),
                triggers=llm_method.get('trigger', ''),
                user_messages=[e['message'] for e in method_info.get('raises_errors', [])],
            )
            
            model_analysis.methods[mname] = method_analysis
            
        # Запази
        self.all_models[model_name] = model_analysis
        
    def _prepare_code_for_llm(self, class_info: dict) -> str:
        """Подготвя код за LLM анализ"""
        
        lines = []
        
        # Клас header
        lines.append(f"class {class_info['name']}:")
        if class_info.get('odoo_model'):
            lines.append(f"    _name = '{class_info['odoo_model']}'")
        if class_info.get('odoo_inherit'):
            lines.append(f"    _inherit = {class_info['odoo_inherit']}")
        if class_info.get('odoo_description'):
            lines.append(f"    _description = '{class_info['odoo_description']}'")
            
        lines.append("")
        
        # Полета
        for field in class_info.get('fields', [])[:20]:  # Лимит
            parts = [f"    {field['name']} = fields.{field['type']}("]
            if field.get('string'):
                parts.append(f"string='{field['string']}'")
            if field.get('help'):
                parts.append(f"help='{field['help'][:100]}'")
            if field.get('comodel_name'):
                parts.append(f"comodel_name='{field['comodel_name']}'")
            if field.get('selection'):
                sel_str = str(field['selection'][:5])
                parts.append(f"selection={sel_str}")
            lines.append(", ".join(parts) + ")")
            
        lines.append("")
        
        # Методи (само сигнатури и docstrings)
        for method in class_info.get('methods', [])[:15]:
            if method['name'].startswith('_') and method['name'] not in ('__init__', '_compute', '_onchange'):
                continue
            decs = "".join(f"    @{d}\n" for d in method.get('decorators', []))
            args = ", ".join(method.get('args', ['self'])[:5])
            lines.append(f"{decs}    def {method['name']}({args}):")
            if method.get('docstring'):
                lines.append(f'        """{method["docstring"][:200]}"""')
            lines.append("        ...")
            lines.append("")
            
        return "\n".join(lines)
        
    def load_po_translations(self, paths: list):
        """Зарежда преводи от PO файлове"""
        
        print(f"\n{'=' * 60}")
        print("ЗАРЕЖДАНЕ НА PO ПРЕВОДИ")
        print(f"{'=' * 60}")
        
        for base_path, _ in paths:
            path = Path(base_path)
            if not path.exists():
                continue
                
            for po_file in path.rglob("bg.po"):
                translations = self._parse_po(str(po_file))
                self.all_translations.update(translations)
                
            for po_file in path.rglob("bg_BG.po"):
                translations = self._parse_po(str(po_file))
                self.all_translations.update(translations)
                
        print(f"✓ Заредени преводи: {len(self.all_translations)}")
        
    def _parse_po(self, filepath: str) -> dict:
        """Парсва PO файл"""
        translations = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return {}
            
        # Извлечи msgid/msgstr двойки
        pattern = r'msgid\s+"((?:[^"\\]|\\.)*)"\s*\nmsgstr\s+"((?:[^"\\]|\\.)*)"'
        
        for match in re.finditer(pattern, content):
            source = match.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
            target = match.group(2).replace('\\n', '\n').replace('\\"', '"').strip()
            
            if source and target and source != target and len(source) < 500:
                translations[source] = target
                
        return translations
        
    def generate_training_data(self, output_dir: str):
        """Генерира training data с богат контекст"""
        
        print(f"\n{'=' * 60}")
        print("ГЕНЕРИРАНЕ НА TRAINING DATA")
        print(f"{'=' * 60}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        samples = []
        seen = set()
        
        for model_name, model in self.all_models.items():
            for field_name, field in model.fields.items():
                if not field.string:
                    continue
                    
                source = field.string
                target = self.all_translations.get(source)
                
                if not target:
                    continue
                    
                # Уникален ключ
                key = hashlib.md5(f"{source}|{target}|{model_name}".encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                
                # Богат контекст
                context = TranslationContext(
                    source_string=source,
                    string_type='field_label',
                    model_name=model.technical_name,
                    model_purpose=model.purpose,
                    field_name=field_name,
                    field_purpose=field.purpose,
                    method_name=None,
                    business_domain=model.business_domain,
                    business_process=model.business_process,
                    user_action=field.data_meaning,
                    semantic_meaning=field.purpose,
                    translation_hints=field.translation_hints,
                    existing_translation=target,
                )
                
                sample = self._create_training_sample(context)
                samples.append(sample)
                
                # Help text
                if field.help:
                    help_target = self.all_translations.get(field.help)
                    if help_target:
                        key2 = hashlib.md5(f"{field.help}|{help_target}".encode()).hexdigest()
                        if key2 not in seen:
                            seen.add(key2)
                            context2 = TranslationContext(
                                source_string=field.help,
                                string_type='help_text',
                                model_name=model.technical_name,
                                model_purpose=model.purpose,
                                field_name=field_name,
                                field_purpose=field.purpose,
                                method_name=None,
                                business_domain=model.business_domain,
                                business_process=model.business_process,
                                user_action="",
                                semantic_meaning=f"Помощен текст за {field.string}",
                                translation_hints="",
                                existing_translation=help_target,
                            )
                            samples.append(self._create_training_sample(context2))
                            
                # Selection values
                if field.selection:
                    for sel in field.selection:
                        sel_label = sel.get('label') if isinstance(sel, dict) else sel[1] if isinstance(sel, (list, tuple)) else None
                        if sel_label:
                            sel_target = self.all_translations.get(sel_label)
                            if sel_target:
                                key3 = hashlib.md5(f"{sel_label}|{sel_target}".encode()).hexdigest()
                                if key3 not in seen:
                                    seen.add(key3)
                                    context3 = TranslationContext(
                                        source_string=sel_label,
                                        string_type='selection_value',
                                        model_name=model.technical_name,
                                        model_purpose=model.purpose,
                                        field_name=field_name,
                                        field_purpose=f"Стойност от списък за {field.string}",
                                        method_name=None,
                                        business_domain=model.business_domain,
                                        business_process=model.business_process,
                                        user_action="",
                                        semantic_meaning="",
                                        translation_hints="",
                                        existing_translation=sel_target,
                                    )
                                    samples.append(self._create_training_sample(context3))
                                    
            # Error messages от методи
            for method_name, method in model.methods.items():
                for msg in method.user_messages:
                    msg_target = self.all_translations.get(msg)
                    if msg_target:
                        key4 = hashlib.md5(f"{msg}|{msg_target}".encode()).hexdigest()
                        if key4 not in seen:
                            seen.add(key4)
                            context4 = TranslationContext(
                                source_string=msg,
                                string_type='error_message',
                                model_name=model.technical_name,
                                model_purpose=model.purpose,
                                field_name=None,
                                field_purpose="",
                                method_name=method_name,
                                business_domain=model.business_domain,
                                business_process=model.business_process,
                                user_action=method.purpose,
                                semantic_meaning=f"Съобщение от {method_name}",
                                translation_hints="",
                                existing_translation=msg_target,
                            )
                            samples.append(self._create_training_sample(context4))
                            
        print(f"✓ Генерирани samples: {len(samples)}")
        
        # Split и запис
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
                
        print(f"✓ Train: {len(train)}, Val: {len(val)}")
        print(f"✓ Записано в: {output_dir}/")
        
    def _create_training_sample(self, ctx: TranslationContext) -> dict:
        """Създава training sample с богат контекст"""
        
        system_prompt = """Ти си експертен преводач за Odoo ERP система. Превеждаш от английски на български.

Правила:
1. Използвай установената Odoo терминология
2. Съобразявай се с бизнес контекста
3. Преводът да е естествен и професионален
4. Отговаряй САМО с превода"""

        # Богат user prompt
        user_parts = [f'Преведи: "{ctx.source_string}"']
        
        context_lines = []
        if ctx.model_name:
            context_lines.append(f"Модел: {ctx.model_name}")
        if ctx.model_purpose:
            context_lines.append(f"Предназначение: {ctx.model_purpose}")
        if ctx.business_domain:
            context_lines.append(f"Домейн: {ctx.business_domain}")
        if ctx.business_process:
            context_lines.append(f"Процес: {ctx.business_process}")
        if ctx.field_name:
            context_lines.append(f"Поле: {ctx.field_name}")
        if ctx.field_purpose:
            context_lines.append(f"Поле означава: {ctx.field_purpose}")
        if ctx.string_type:
            type_names = {
                'field_label': 'Етикет на поле',
                'help_text': 'Помощен текст',
                'selection_value': 'Стойност от списък',
                'error_message': 'Съобщение за грешка',
                'button_label': 'Етикет на бутон',
            }
            context_lines.append(f"Тип: {type_names.get(ctx.string_type, ctx.string_type)}")
        if ctx.translation_hints:
            context_lines.append(f"Подсказка: {ctx.translation_hints}")
            
        if context_lines:
            user_parts.append("\nКонтекст:")
            user_parts.extend(f"  {line}" for line in context_lines)
            
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_parts)},
                {"role": "assistant", "content": ctx.existing_translation}
            ]
        }
        
    def save_analysis(self, output_dir: str):
        """Запазва пълния анализ"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Модели
        models_data = {}
        for name, model in self.all_models.items():
            models_data[name] = {
                'name': model.name,
                'technical_name': model.technical_name,
                'purpose': model.purpose,
                'business_domain': model.business_domain,
                'business_process': model.business_process,
                'user_role': model.user_role,
                'module': model.module,
                'fields': {
                    fname: {
                        'string': f.string,
                        'purpose': f.purpose,
                        'business_context': f.business_context,
                        'data_meaning': f.data_meaning,
                        'translation_hints': f.translation_hints,
                    }
                    for fname, f in model.fields.items()
                }
            }
            
        with open(output_path / 'deep_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(models_data, f, ensure_ascii=False, indent=2)
            
        print(f"✓ Анализ запазен в: {output_dir}/deep_analysis.json")


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Semantic Odoo Analyzer')
    parser.add_argument('--model', '-m', default='qwen2.5:14b',
                       help='Ollama модел за анализ')
    parser.add_argument('--output', '-o', default='./training_data',
                       help='Изходна директория')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  DEEP SEMANTIC ODOO ANALYZER")
    print("=" * 60)
    
    # Инициализирай LLM
    CONFIG["llm_model"] = args.model
    llm = OllamaClient(CONFIG["llm_base_url"], CONFIG["llm_model"])
    
    # Създай анализатор
    analyzer = DeepSemanticAnalyzer(llm)
    
    # Зареди преводи първо
    analyzer.load_po_translations(CONFIG["odoo_paths"])
    
    # Анализирай всяка директория
    for path, source_type in CONFIG["odoo_paths"]:
        if Path(path).exists():
            analyzer.analyze_directory(path, source_type)
            
    # Запази анализа
    analyzer.save_analysis(CONFIG["output_dir"])
    
    # Генерирай training data
    analyzer.generate_training_data(args.output)
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == '__main__':
    main()
