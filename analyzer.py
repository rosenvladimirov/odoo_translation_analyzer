#!/usr/bin/env python3
"""
Odoo Source Code Analyzer for Translation Context
==================================================
Анализира Odoo source code за да извлече:
1. Модели и тяхното предназначение
2. Полета с пълен контекст
3. Бизнес логика и workflow
4. Релации между модели
5. Готови данни за обучение на translation модел
"""

import ast
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict
import xml.etree.ElementTree as ET


@dataclass
class FieldInfo:
    """Информация за поле в модел"""
    name: str
    field_type: str
    string: Optional[str] = None
    help: Optional[str] = None
    related: Optional[str] = None
    comodel_name: Optional[str] = None
    selection: Optional[list] = None
    compute: Optional[str] = None
    inverse: Optional[str] = None
    default: Optional[str] = None
    required: bool = False
    readonly: bool = False
    store: bool = True
    groups: Optional[str] = None
    # Анализ
    purpose: Optional[str] = None  # Какво прави полето
    usage_context: list = field(default_factory=list)  # Къде се използва


@dataclass
class MethodInfo:
    """Информация за метод"""
    name: str
    decorators: list
    docstring: Optional[str] = None
    fields_accessed: list = field(default_factory=list)
    fields_written: list = field(default_factory=list)
    calls_methods: list = field(default_factory=list)
    purpose: Optional[str] = None


@dataclass
class ModelInfo:
    """Пълна информация за модел"""
    name: str
    inherit: list
    description: Optional[str] = None
    module: str = ""
    file_path: str = ""
    
    fields: dict = field(default_factory=dict)  # name -> FieldInfo
    methods: dict = field(default_factory=dict)  # name -> MethodInfo
    
    # Анализ
    purpose: Optional[str] = None
    business_domain: Optional[str] = None
    related_models: list = field(default_factory=list)
    workflow_states: list = field(default_factory=list)


class OdooFieldVisitor(ast.NodeVisitor):
    """AST Visitor за извличане на Odoo полета и методи"""
    
    FIELD_TYPES = {
        'Char', 'Text', 'Html', 'Integer', 'Float', 'Monetary',
        'Boolean', 'Date', 'Datetime', 'Binary', 'Image',
        'Selection', 'Many2one', 'One2many', 'Many2many',
        'Reference', 'Properties', 'PropertiesDefinition'
    }
    
    def __init__(self):
        self.fields = {}
        self.methods = {}
        self.model_name = None
        self.model_inherit = []
        self.model_description = None
        self.current_class = None
        
    def visit_ClassDef(self, node):
        """Посещава class дефиниция"""
        self.current_class = node.name
        
        # Търси _name, _inherit, _description в class body
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id == '_name':
                            self.model_name = self._get_string_value(item.value)
                        elif target.id == '_inherit':
                            self.model_inherit = self._get_inherit_value(item.value)
                        elif target.id == '_description':
                            self.model_description = self._get_string_value(item.value)
        
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        """Посещава присвоявания (за полета)"""
        if not self.current_class:
            return
            
        for target in node.targets:
            if isinstance(target, ast.Name):
                field_info = self._parse_field(target.id, node.value)
                if field_info:
                    self.fields[target.id] = field_info
                    
    def visit_FunctionDef(self, node):
        """Посещава функции/методи"""
        if not self.current_class:
            return
            
        method_info = self._parse_method(node)
        if method_info:
            self.methods[node.name] = method_info
            
    def _parse_field(self, name: str, value_node) -> Optional[FieldInfo]:
        """Парсва дефиниция на поле"""
        if not isinstance(value_node, ast.Call):
            return None
            
        # Определя типа на полето
        field_type = None
        if isinstance(value_node.func, ast.Attribute):
            if value_node.func.attr in self.FIELD_TYPES:
                field_type = value_node.func.attr
        elif isinstance(value_node.func, ast.Name):
            if value_node.func.id in self.FIELD_TYPES:
                field_type = value_node.func.id
                
        if not field_type:
            return None
            
        # Извлича атрибути на полето
        field_info = FieldInfo(name=name, field_type=field_type)
        
        for keyword in value_node.keywords:
            key = keyword.arg
            val = self._get_value(keyword.value)
            
            if key == 'string':
                field_info.string = val
            elif key == 'help':
                field_info.help = val
            elif key == 'related':
                field_info.related = val
            elif key == 'comodel_name':
                field_info.comodel_name = val
            elif key == 'selection':
                field_info.selection = self._parse_selection(keyword.value)
            elif key == 'compute':
                field_info.compute = val
            elif key == 'inverse':
                field_info.inverse = val
            elif key == 'default':
                field_info.default = str(val)
            elif key == 'required':
                field_info.required = val is True
            elif key == 'readonly':
                field_info.readonly = val is True
            elif key == 'store':
                field_info.store = val is not False
            elif key == 'groups':
                field_info.groups = val
                
        return field_info
        
    def _parse_method(self, node: ast.FunctionDef) -> MethodInfo:
        """Парсва метод"""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)
                elif isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                    
        # Извлича docstring
        docstring = ast.get_docstring(node)
        
        # Анализира тялото за достъп до полета
        field_access = FieldAccessAnalyzer()
        field_access.visit(node)
        
        return MethodInfo(
            name=node.name,
            decorators=decorators,
            docstring=docstring,
            fields_accessed=list(field_access.read_fields),
            fields_written=list(field_access.write_fields),
            calls_methods=list(field_access.method_calls)
        )
        
    def _parse_selection(self, node) -> list:
        """Парсва selection стойности"""
        selections = []
        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) >= 2:
                    key = self._get_string_value(item.elts[0])
                    label = self._get_string_value(item.elts[1])
                    if key and label:
                        selections.append((key, label))
        return selections
        
    def _get_string_value(self, node) -> Optional[str]:
        """Извлича string стойност от AST node"""
        if isinstance(node, ast.Constant):
            return str(node.value) if node.value else None
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        return None
        
    def _get_inherit_value(self, node) -> list:
        """Извлича _inherit стойност"""
        if isinstance(node, ast.Constant):
            return [node.value] if node.value else []
        elif isinstance(node, ast.Str):
            return [node.s]
        elif isinstance(node, ast.List):
            return [self._get_string_value(el) for el in node.elts if self._get_string_value(el)]
        return []
        
    def _get_value(self, node):
        """Извлича произволна стойност"""
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
        return None


class FieldAccessAnalyzer(ast.NodeVisitor):
    """Анализира достъпа до полета в метод"""
    
    def __init__(self):
        self.read_fields = set()
        self.write_fields = set()
        self.method_calls = set()
        
    def visit_Attribute(self, node):
        """Открива self.field_name достъпи"""
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            # Проверява дали е write или read
            parent = getattr(node, '_parent', None)
            if isinstance(parent, ast.Assign):
                self.write_fields.add(node.attr)
            else:
                self.read_fields.add(node.attr)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """Открива извиквания на методи"""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                self.method_calls.add(node.func.attr)
        self.generic_visit(node)


class OdooXMLAnalyzer:
    """Анализира XML файлове за views, actions, menus"""
    
    def __init__(self):
        self.views = []
        self.actions = []
        self.menus = []
        self.field_labels = {}  # field_name -> list of labels from views
        
    def parse_file(self, filepath: str):
        """Парсва XML файл"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            self._process_element(root, filepath)
        except ET.ParseError as e:
            print(f"XML Parse Error in {filepath}: {e}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    def _process_element(self, element, filepath: str):
        """Рекурсивно обработва XML елементи"""
        # Views
        if element.tag == 'record' and element.get('model') in ('ir.ui.view',):
            self._process_view(element, filepath)
            
        # Actions
        if element.tag == 'record' and element.get('model') in ('ir.actions.act_window',):
            self._process_action(element, filepath)
            
        # Menus
        if element.tag == 'menuitem':
            self._process_menu(element, filepath)
            
        # Field usage in views
        if element.tag == 'field' and element.get('name'):
            field_name = element.get('name')
            string = element.get('string')
            if string:
                if field_name not in self.field_labels:
                    self.field_labels[field_name] = []
                self.field_labels[field_name].append({
                    'label': string,
                    'file': filepath
                })
                
        for child in element:
            self._process_element(child, filepath)
            
    def _process_view(self, element, filepath):
        """Обработва view record"""
        view_info = {
            'id': element.get('id'),
            'file': filepath
        }
        for field in element.findall('.//field'):
            name = field.get('name')
            if name == 'name':
                view_info['name'] = field.text
            elif name == 'model':
                view_info['model'] = field.text
            elif name == 'arch':
                # Анализира view архитектурата
                view_info['fields_used'] = self._extract_view_fields(field)
        self.views.append(view_info)
        
    def _process_action(self, element, filepath):
        """Обработва action record"""
        action_info = {
            'id': element.get('id'),
            'file': filepath
        }
        for field in element.findall('.//field'):
            name = field.get('name')
            if name == 'name':
                action_info['name'] = field.text
            elif name == 'res_model':
                action_info['model'] = field.text
        self.actions.append(action_info)
        
    def _process_menu(self, element, filepath):
        """Обработва menuitem"""
        self.menus.append({
            'id': element.get('id'),
            'name': element.get('name'),
            'action': element.get('action'),
            'parent': element.get('parent'),
            'file': filepath
        })
        
    def _extract_view_fields(self, arch_element) -> list:
        """Извлича използвани полета от view"""
        fields = []
        # Парсва вътрешния XML ако е текст
        if arch_element.text:
            try:
                inner = ET.fromstring(f"<root>{arch_element.text}</root>")
                for field in inner.iter('field'):
                    fields.append({
                        'name': field.get('name'),
                        'string': field.get('string'),
                        'widget': field.get('widget'),
                        'invisible': field.get('invisible'),
                        'readonly': field.get('readonly')
                    })
            except:
                pass
        return fields


class OdooPOAnalyzer:
    """Анализира PO файлове за съществуващи преводи"""
    
    def __init__(self):
        self.translations = {}  # source -> {lang: translation}
        self.contexts = {}  # source -> list of contexts
        
    def parse_file(self, filepath: str, lang: str = 'bg'):
        """Парсва PO файл"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Прост PO parser
            entries = re.findall(
                r'#:\s*([^\n]+)\n(?:#,\s*[^\n]+\n)?msgid\s+"([^"]+)"\nmsgstr\s+"([^"]*)"',
                content
            )
            
            for context, source, translation in entries:
                if source not in self.translations:
                    self.translations[source] = {}
                    self.contexts[source] = []
                    
                self.translations[source][lang] = translation
                self.contexts[source].append(context)
                
        except Exception as e:
            print(f"Error parsing PO file {filepath}: {e}")


class OdooSourceAnalyzer:
    """Главен анализатор на Odoo source code"""
    
    def __init__(self, odoo_path: str):
        self.odoo_path = Path(odoo_path)
        self.models = {}  # model_name -> ModelInfo
        self.xml_analyzer = OdooXMLAnalyzer()
        self.po_analyzer = OdooPOAnalyzer()
        
    def analyze(self, modules: list = None):
        """Пълен анализ на source code"""
        print("=== Odoo Source Code Analyzer ===\n")
        
        # 1. Намери всички модули
        if modules:
            module_paths = [self.odoo_path / m for m in modules]
        else:
            module_paths = self._find_modules()
            
        print(f"Намерени модули: {len(module_paths)}\n")
        
        # 2. Анализирай Python файлове
        print("Анализиране на Python файлове...")
        for module_path in module_paths:
            self._analyze_module(module_path)
            
        print(f"Намерени модели: {len(self.models)}\n")
        
        # 3. Анализирай XML файлове
        print("Анализиране на XML файлове...")
        for module_path in module_paths:
            self._analyze_xml(module_path)
            
        # 4. Анализирай PO файлове
        print("Анализиране на преводи...")
        for module_path in module_paths:
            self._analyze_po(module_path)
            
        # 5. Обогати с контекст
        print("Генериране на контекст...")
        self._enrich_context()
        
        return self
        
    def _find_modules(self) -> list:
        """Намира всички Odoo модули"""
        modules = []
        
        # addons директория
        addons_path = self.odoo_path / 'addons'
        if addons_path.exists():
            for item in addons_path.iterdir():
                if item.is_dir() and (item / '__manifest__.py').exists():
                    modules.append(item)
                    
        # odoo/addons директория
        odoo_addons = self.odoo_path / 'odoo' / 'addons'
        if odoo_addons.exists():
            for item in odoo_addons.iterdir():
                if item.is_dir() and (item / '__manifest__.py').exists():
                    modules.append(item)
                    
        return modules
        
    def _analyze_module(self, module_path: Path):
        """Анализира един модул"""
        module_name = module_path.name
        
        # Намери всички Python файлове
        for py_file in module_path.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                tree = ast.parse(source)
                
                # Анализирай за модели
                visitor = OdooFieldVisitor()
                visitor.visit(tree)
                
                if visitor.model_name or visitor.model_inherit:
                    model_name = visitor.model_name or (
                        visitor.model_inherit[0] if visitor.model_inherit else None
                    )
                    
                    if model_name:
                        if model_name not in self.models:
                            self.models[model_name] = ModelInfo(
                                name=model_name,
                                inherit=visitor.model_inherit,
                                description=visitor.model_description,
                                module=module_name,
                                file_path=str(py_file)
                            )
                        else:
                            # Merge с съществуващ модел (extension)
                            existing = self.models[model_name]
                            existing.inherit.extend(visitor.model_inherit)
                            
                        # Добави полета и методи
                        model = self.models[model_name]
                        for fname, finfo in visitor.fields.items():
                            model.fields[fname] = finfo
                        for mname, minfo in visitor.methods.items():
                            model.methods[mname] = minfo
                            
            except SyntaxError as e:
                print(f"  Syntax error in {py_file}: {e}")
            except Exception as e:
                print(f"  Error in {py_file}: {e}")
                
    def _analyze_xml(self, module_path: Path):
        """Анализира XML файловете на модул"""
        for xml_file in module_path.rglob('*.xml'):
            self.xml_analyzer.parse_file(str(xml_file))
            
    def _analyze_po(self, module_path: Path):
        """Анализира PO файловете на модул"""
        i18n_path = module_path / 'i18n'
        if i18n_path.exists():
            for po_file in i18n_path.glob('*.po'):
                lang = po_file.stem
                self.po_analyzer.parse_file(str(po_file), lang)
                
    def _enrich_context(self):
        """Обогатява моделите с контекстна информация"""
        
        # Определя business domain по модел name
        domain_keywords = {
            'sale': 'Продажби',
            'purchase': 'Покупки', 
            'stock': 'Склад',
            'account': 'Счетоводство',
            'hr': 'Човешки ресурси',
            'mrp': 'Производство',
            'project': 'Проекти',
            'crm': 'CRM',
            'website': 'Уебсайт',
            'pos': 'Точка на продажба',
            'fleet': 'Автопарк',
            'maintenance': 'Поддръжка',
            'helpdesk': 'Поддръжка клиенти',
            'mail': 'Съобщения',
            'calendar': 'Календар'
        }
        
        for model_name, model in self.models.items():
            # Business domain
            for keyword, domain in domain_keywords.items():
                if keyword in model_name.lower() or keyword in model.module.lower():
                    model.business_domain = domain
                    break
                    
            # Related models от Many2one/One2many/Many2many
            for field in model.fields.values():
                if field.comodel_name and field.comodel_name not in model.related_models:
                    model.related_models.append(field.comodel_name)
                    
            # Workflow states от selection полета с name='state'
            if 'state' in model.fields:
                state_field = model.fields['state']
                if state_field.selection:
                    model.workflow_states = state_field.selection
                    
            # Field labels от XML views
            for fname, finfo in model.fields.items():
                if fname in self.xml_analyzer.field_labels:
                    finfo.usage_context = self.xml_analyzer.field_labels[fname]
                    
    def generate_training_data(self, output_path: str = 'training_data.jsonl'):
        """Генерира training dataset за translation модел"""
        
        training_samples = []
        
        for model_name, model in self.models.items():
            # Model description
            if model.description:
                sample = {
                    'type': 'model_description',
                    'model': model_name,
                    'module': model.module,
                    'business_domain': model.business_domain,
                    'source': model.description,
                    'context': f"Description of Odoo model {model_name} in {model.business_domain or 'general'} domain",
                    'existing_translation': self.po_analyzer.translations.get(model.description, {})
                }
                training_samples.append(sample)
                
            # Field strings
            for fname, finfo in model.fields.items():
                if finfo.string:
                    sample = {
                        'type': 'field_label',
                        'model': model_name,
                        'field': fname,
                        'field_type': finfo.field_type,
                        'module': model.module,
                        'business_domain': model.business_domain,
                        'source': finfo.string,
                        'help': finfo.help,
                        'context': self._build_field_context(model, finfo),
                        'related_model': finfo.comodel_name,
                        'selection_values': finfo.selection,
                        'existing_translation': self.po_analyzer.translations.get(finfo.string, {}),
                        'usage_in_views': finfo.usage_context
                    }
                    training_samples.append(sample)
                    
                # Field help
                if finfo.help:
                    sample = {
                        'type': 'field_help',
                        'model': model_name,
                        'field': fname,
                        'module': model.module,
                        'business_domain': model.business_domain,
                        'source': finfo.help,
                        'field_label': finfo.string,
                        'context': f"Help text for field '{fname}' ({finfo.string}) in {model_name}",
                        'existing_translation': self.po_analyzer.translations.get(finfo.help, {})
                    }
                    training_samples.append(sample)
                    
                # Selection values
                if finfo.selection:
                    for key, label in finfo.selection:
                        sample = {
                            'type': 'selection_value',
                            'model': model_name,
                            'field': fname,
                            'selection_key': key,
                            'module': model.module,
                            'business_domain': model.business_domain,
                            'source': label,
                            'context': f"Selection value '{key}' for field '{fname}' in {model_name}. "
                                      f"Other values: {[k for k, v in finfo.selection if k != key]}",
                            'existing_translation': self.po_analyzer.translations.get(label, {})
                        }
                        training_samples.append(sample)
                        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
        print(f"\nГенерирани {len(training_samples)} training samples в {output_path}")
        
        return training_samples
        
    def _build_field_context(self, model: ModelInfo, field: FieldInfo) -> str:
        """Генерира контекстно описание за поле"""
        parts = []
        
        # Тип на полето
        type_descriptions = {
            'Char': 'text field',
            'Text': 'multi-line text field',
            'Html': 'HTML content field',
            'Integer': 'integer number field',
            'Float': 'decimal number field',
            'Monetary': 'monetary/currency field',
            'Boolean': 'yes/no checkbox field',
            'Date': 'date field',
            'Datetime': 'date and time field',
            'Binary': 'file/binary field',
            'Image': 'image field',
            'Selection': 'dropdown selection field',
            'Many2one': 'reference to another record',
            'One2many': 'list of related records',
            'Many2many': 'multiple selection of records'
        }
        
        parts.append(f"A {type_descriptions.get(field.field_type, field.field_type)}")
        parts.append(f"in Odoo model '{model.name}'")
        
        if model.business_domain:
            parts.append(f"({model.business_domain} domain)")
            
        if field.comodel_name:
            parts.append(f"linked to '{field.comodel_name}'")
            
        if field.required:
            parts.append("(required)")
            
        if field.readonly:
            parts.append("(read-only)")
            
        if field.compute:
            parts.append("(computed field)")
            
        return ' '.join(parts)
        
    def generate_summary(self) -> dict:
        """Генерира обобщена статистика"""
        total_fields = sum(len(m.fields) for m in self.models.values())
        total_methods = sum(len(m.methods) for m in self.models.values())
        
        domains = defaultdict(int)
        for model in self.models.values():
            if model.business_domain:
                domains[model.business_domain] += 1
                
        return {
            'total_models': len(self.models),
            'total_fields': total_fields,
            'total_methods': total_methods,
            'total_views': len(self.xml_analyzer.views),
            'total_actions': len(self.xml_analyzer.actions),
            'total_menus': len(self.xml_analyzer.menus),
            'total_translations': len(self.po_analyzer.translations),
            'models_by_domain': dict(domains)
        }
        
    def export_models(self, output_path: str = 'models_analysis.json'):
        """Експортира пълния анализ на моделите"""
        
        def model_to_dict(model: ModelInfo) -> dict:
            return {
                'name': model.name,
                'inherit': model.inherit,
                'description': model.description,
                'module': model.module,
                'file_path': model.file_path,
                'business_domain': model.business_domain,
                'related_models': model.related_models,
                'workflow_states': model.workflow_states,
                'fields': {k: asdict(v) for k, v in model.fields.items()},
                'methods': {k: asdict(v) for k, v in model.methods.items()}
            }
            
        data = {
            'summary': self.generate_summary(),
            'models': {k: model_to_dict(v) for k, v in self.models.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Експортиран анализ в {output_path}")


# Примерно използване
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py /path/to/odoo [module1,module2,...]")
        sys.exit(1)
        
    odoo_path = sys.argv[1]
    modules = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    
    analyzer = OdooSourceAnalyzer(odoo_path)
    analyzer.analyze(modules)
    
    # Покажи обобщение
    summary = analyzer.generate_summary()
    print("\n=== Обобщение ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
        
    # Генерирай training data
    analyzer.generate_training_data('training_data.jsonl')
    
    # Експортирай пълен анализ
    analyzer.export_models('models_analysis.json')
