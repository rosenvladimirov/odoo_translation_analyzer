#!/usr/bin/env python3
"""
Semantic Analyzer - Дълбок анализ на Odoo код
=============================================
Анализира бизнес логиката и семантиката на кода
за по-добър контекст при превод.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class SemanticContext:
    """Семантичен контекст за string"""
    source_string: str
    translation_type: str  # field_label, help_text, error_message, button_label, etc.
    
    # Технически контекст
    model_name: str
    field_name: Optional[str] = None
    method_name: Optional[str] = None
    
    # Бизнес контекст
    business_domain: str = ""
    business_process: str = ""  # напр. "order confirmation", "invoice creation"
    
    # Семантичен контекст
    action_type: str = ""  # create, update, delete, compute, validate
    data_type: str = ""  # date, money, quantity, reference, status
    user_role: str = ""  # sales_person, accountant, manager
    
    # Допълнителен контекст
    related_terms: list = field(default_factory=list)
    similar_fields: list = field(default_factory=list)
    workflow_stage: str = ""
    
    # Съществуващи преводи
    existing_translations: dict = field(default_factory=dict)
    
    def to_prompt(self) -> str:
        """Генерира prompt за translation модел"""
        parts = [f"Translate to Bulgarian: \"{self.source_string}\""]
        parts.append(f"\nContext:")
        parts.append(f"- Type: {self.translation_type}")
        parts.append(f"- Odoo Model: {self.model_name}")
        
        if self.field_name:
            parts.append(f"- Field: {self.field_name}")
        if self.business_domain:
            parts.append(f"- Business Domain: {self.business_domain}")
        if self.business_process:
            parts.append(f"- Process: {self.business_process}")
        if self.action_type:
            parts.append(f"- Action: {self.action_type}")
        if self.data_type:
            parts.append(f"- Data Type: {self.data_type}")
        if self.workflow_stage:
            parts.append(f"- Workflow Stage: {self.workflow_stage}")
        if self.related_terms:
            parts.append(f"- Related Terms: {', '.join(self.related_terms)}")
        if self.existing_translations:
            parts.append(f"- Existing Translations: {self.existing_translations}")
            
        return '\n'.join(parts)


class SemanticAnalyzer:
    """Анализатор за извличане на семантичен контекст"""
    
    # Mapping на Odoo модели към бизнес домейни
    DOMAIN_MAPPING = {
        'sale.order': ('Продажби', 'Поръчка за продажба'),
        'sale.order.line': ('Продажби', 'Ред от поръчка'),
        'purchase.order': ('Покупки', 'Поръчка за покупка'),
        'purchase.order.line': ('Покупки', 'Ред от поръчка за покупка'),
        'account.move': ('Счетоводство', 'Счетоводен документ/Фактура'),
        'account.move.line': ('Счетоводство', 'Счетоводен запис'),
        'account.payment': ('Счетоводство', 'Плащане'),
        'account.account': ('Счетоводство', 'Сметка'),
        'account.journal': ('Счетоводство', 'Дневник'),
        'account.tax': ('Счетоводство', 'Данък'),
        'stock.picking': ('Склад', 'Складова операция'),
        'stock.move': ('Склад', 'Складово движение'),
        'stock.quant': ('Склад', 'Наличност'),
        'stock.warehouse': ('Склад', 'Склад'),
        'stock.location': ('Склад', 'Местоположение'),
        'product.product': ('Продукти', 'Продукт'),
        'product.template': ('Продукти', 'Шаблон на продукт'),
        'product.category': ('Продукти', 'Категория'),
        'res.partner': ('Контакти', 'Партньор/Контакт'),
        'res.users': ('Система', 'Потребител'),
        'res.company': ('Система', 'Фирма'),
        'hr.employee': ('HR', 'Служител'),
        'hr.contract': ('HR', 'Трудов договор'),
        'hr.payslip': ('HR', 'Фиш за заплата'),
        'hr.leave': ('HR', 'Отпуска'),
        'mrp.production': ('Производство', 'Производствена поръчка'),
        'mrp.bom': ('Производство', 'Спецификация'),
        'mrp.workorder': ('Производство', 'Работна поръчка'),
        'project.project': ('Проекти', 'Проект'),
        'project.task': ('Проекти', 'Задача'),
        'crm.lead': ('CRM', 'Потенциален клиент'),
        'crm.team': ('CRM', 'Търговски екип'),
    }
    
    # Patterns за разпознаване на типове данни
    DATA_TYPE_PATTERNS = {
        'date': ['date', 'created', 'updated', 'deadline', 'scheduled', 'validity'],
        'money': ['amount', 'price', 'cost', 'total', 'subtotal', 'tax', 'discount', 'balance', 'debit', 'credit'],
        'quantity': ['qty', 'quantity', 'count', 'number', 'stock', 'available'],
        'reference': ['ref', 'reference', 'number', 'code', 'name', 'title'],
        'status': ['state', 'status', 'stage', 'type'],
        'percentage': ['percent', 'rate', 'ratio', 'margin'],
        'contact': ['email', 'phone', 'mobile', 'address', 'street', 'city', 'country'],
        'document': ['document', 'attachment', 'file', 'report'],
    }
    
    # Patterns за разпознаване на действия
    ACTION_PATTERNS = {
        'create': ['create', 'new', 'add', 'generate', 'draft'],
        'update': ['update', 'edit', 'modify', 'change', 'set'],
        'delete': ['delete', 'remove', 'cancel', 'unlink'],
        'compute': ['compute', 'calculate', 'total', 'sum', 'count'],
        'validate': ['validate', 'confirm', 'approve', 'check', 'verify'],
        'send': ['send', 'post', 'email', 'notify', 'share'],
        'print': ['print', 'report', 'export', 'download'],
    }
    
    # Workflow stages
    WORKFLOW_STAGES = {
        'draft': 'Чернова',
        'sent': 'Изпратено',
        'confirmed': 'Потвърдено',
        'done': 'Приключено',
        'cancel': 'Отменено',
        'posted': 'Осчетоводено',
        'paid': 'Платено',
        'approved': 'Одобрено',
        'refused': 'Отказано',
        'open': 'Отворено',
        'closed': 'Затворено',
    }
    
    def __init__(self, models_data: dict):
        """
        models_data: dict от model_name -> ModelInfo
        """
        self.models = models_data
        self.field_term_index = self._build_term_index()
        
    def _build_term_index(self) -> dict:
        """Индексира термини от всички полета за намиране на свързани"""
        index = defaultdict(list)
        
        for model_name, model in self.models.items():
            for field_name, field_info in model.get('fields', {}).items():
                # Индексирай по думи от string
                if field_info.get('string'):
                    words = field_info['string'].lower().split()
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            index[word].append({
                                'model': model_name,
                                'field': field_name,
                                'string': field_info['string']
                            })
        return dict(index)
        
    def analyze_field(self, model_name: str, field_name: str, field_info: dict) -> SemanticContext:
        """Анализира поле и генерира семантичен контекст"""
        
        source_string = field_info.get('string', field_name)
        
        ctx = SemanticContext(
            source_string=source_string,
            translation_type='field_label',
            model_name=model_name,
            field_name=field_name
        )
        
        # Business domain
        if model_name in self.DOMAIN_MAPPING:
            ctx.business_domain, ctx.business_process = self.DOMAIN_MAPPING[model_name]
        else:
            # Опит за извличане от model name
            ctx.business_domain = self._infer_domain(model_name)
            
        # Data type
        ctx.data_type = self._detect_data_type(field_name, field_info)
        
        # Workflow stage (ако е state поле)
        if field_name == 'state' and field_info.get('selection'):
            ctx.workflow_stage = 'state_field'
            
        # Related terms
        ctx.related_terms = self._find_related_terms(source_string)
        
        # Similar fields
        ctx.similar_fields = self._find_similar_fields(model_name, field_name, source_string)
        
        return ctx
        
    def analyze_method(self, model_name: str, method_name: str, method_info: dict) -> dict:
        """Анализира метод за разбиране на бизнес логиката"""
        
        analysis = {
            'method': method_name,
            'model': model_name,
            'action_type': self._detect_action_type(method_name),
            'is_compute': 'compute' in method_info.get('decorators', []),
            'is_onchange': 'onchange' in method_info.get('decorators', []),
            'is_constraint': 'constrains' in method_info.get('decorators', []),
            'affects_fields': method_info.get('fields_written', []),
            'reads_fields': method_info.get('fields_accessed', []),
            'purpose': method_info.get('docstring', ''),
        }
        
        # Извлечи бизнес логика от docstring и име
        if method_info.get('docstring'):
            analysis['business_logic'] = self._extract_business_logic(method_info['docstring'])
            
        return analysis
        
    def analyze_selection_value(self, model_name: str, field_name: str, 
                                 key: str, label: str) -> SemanticContext:
        """Анализира стойност от selection поле"""
        
        ctx = SemanticContext(
            source_string=label,
            translation_type='selection_value',
            model_name=model_name,
            field_name=field_name
        )
        
        # Business domain
        if model_name in self.DOMAIN_MAPPING:
            ctx.business_domain, ctx.business_process = self.DOMAIN_MAPPING[model_name]
            
        # Ако е state, добави workflow контекст
        if field_name == 'state':
            ctx.workflow_stage = self.WORKFLOW_STAGES.get(key, key)
            
        return ctx
        
    def analyze_error_message(self, model_name: str, method_name: str, 
                               message: str) -> SemanticContext:
        """Анализира error message"""
        
        ctx = SemanticContext(
            source_string=message,
            translation_type='error_message',
            model_name=model_name,
            method_name=method_name
        )
        
        # Business domain
        if model_name in self.DOMAIN_MAPPING:
            ctx.business_domain, _ = self.DOMAIN_MAPPING[model_name]
            
        # Действие което предизвиква грешката
        ctx.action_type = self._detect_action_type(method_name)
        
        return ctx
        
    def _infer_domain(self, model_name: str) -> str:
        """Извлича бизнес домейн от име на модел"""
        parts = model_name.split('.')
        
        domain_map = {
            'sale': 'Продажби',
            'purchase': 'Покупки',
            'account': 'Счетоводство',
            'stock': 'Склад',
            'product': 'Продукти',
            'res': 'Система',
            'hr': 'HR',
            'mrp': 'Производство',
            'project': 'Проекти',
            'crm': 'CRM',
            'website': 'Уебсайт',
            'pos': 'POS',
            'fleet': 'Автопарк',
            'maintenance': 'Поддръжка',
            'mail': 'Съобщения',
        }
        
        for part in parts:
            if part in domain_map:
                return domain_map[part]
                
        return 'Общ'
        
    def _detect_data_type(self, field_name: str, field_info: dict) -> str:
        """Определя типа данни на поле"""
        
        field_type = field_info.get('field_type', '')
        name_lower = field_name.lower()
        
        # По тип на полето
        if field_type in ('Date', 'Datetime'):
            return 'date'
        if field_type == 'Monetary':
            return 'money'
        if field_type == 'Selection':
            return 'status'
        if field_type in ('Many2one', 'Many2many', 'One2many'):
            return 'reference'
            
        # По име на полето
        for data_type, patterns in self.DATA_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return data_type
                    
        return 'text'
        
    def _detect_action_type(self, method_name: str) -> str:
        """Определя типа действие на метод"""
        
        name_lower = method_name.lower()
        
        for action, patterns in self.ACTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return action
                    
        return 'unknown'
        
    def _find_related_terms(self, source_string: str) -> list:
        """Намира свързани термини"""
        
        related = []
        words = source_string.lower().split()
        
        for word in words:
            if word in self.field_term_index:
                for entry in self.field_term_index[word][:5]:
                    if entry['string'] != source_string:
                        related.append(entry['string'])
                        
        return list(set(related))[:10]
        
    def _find_similar_fields(self, model_name: str, field_name: str, 
                              source_string: str) -> list:
        """Намира подобни полета в други модели"""
        
        similar = []
        
        for other_model, model_data in self.models.items():
            if other_model == model_name:
                continue
                
            for fname, finfo in model_data.get('fields', {}).items():
                # Същото име на поле
                if fname == field_name:
                    similar.append({
                        'model': other_model,
                        'field': fname,
                        'string': finfo.get('string', fname)
                    })
                # Същия string
                elif finfo.get('string') == source_string:
                    similar.append({
                        'model': other_model,
                        'field': fname,
                        'string': source_string
                    })
                    
        return similar[:10]
        
    def _extract_business_logic(self, docstring: str) -> str:
        """Извлича бизнес логика от docstring"""
        
        # Почисти docstring
        lines = docstring.strip().split('\n')
        
        # Първият ред обикновено е summary
        if lines:
            return lines[0].strip()
            
        return ''


class TranslationTerminology:
    """База данни с терминология за последователен превод"""
    
    def __init__(self):
        self.terms = {}
        self.domain_terms = defaultdict(dict)
        
    def add_term(self, source: str, translation: str, domain: str = 'general'):
        """Добавя термин"""
        self.terms[source.lower()] = translation
        self.domain_terms[domain][source.lower()] = translation
        
    def get_translation(self, source: str, domain: str = None) -> Optional[str]:
        """Взима превод ако съществува"""
        source_lower = source.lower()
        
        # Първо провери domain-specific
        if domain and domain in self.domain_terms:
            if source_lower in self.domain_terms[domain]:
                return self.domain_terms[domain][source_lower]
                
        # После general
        return self.terms.get(source_lower)
        
    def load_from_po(self, po_translations: dict):
        """Зарежда терминология от PO файлове"""
        for source, translations in po_translations.items():
            if 'bg' in translations and translations['bg']:
                self.add_term(source, translations['bg'])
                
    def export(self, filepath: str):
        """Експортира терминология"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'general': dict(self.terms),
                'by_domain': {k: dict(v) for k, v in self.domain_terms.items()}
            }, f, ensure_ascii=False, indent=2)


# Стандартна Odoo терминология за български
STANDARD_ODOO_TERMS_BG = {
    # Общи
    'Name': 'Име',
    'Description': 'Описание',
    'Notes': 'Бележки',
    'Active': 'Активен',
    'Sequence': 'Поредност',
    'Company': 'Фирма',
    'Currency': 'Валута',
    'Date': 'Дата',
    'Reference': 'Референция',
    'Tags': 'Етикети',
    
    # Продажби
    'Sales Order': 'Поръчка за продажба',
    'Quotation': 'Оферта',
    'Customer': 'Клиент',
    'Salesperson': 'Търговец',
    'Order Date': 'Дата на поръчка',
    'Delivery Date': 'Дата на доставка',
    'Payment Terms': 'Условия на плащане',
    
    # Покупки
    'Purchase Order': 'Поръчка за покупка',
    'Request for Quotation': 'Запитване за оферта',
    'Vendor': 'Доставчик',
    'Supplier': 'Доставчик',
    
    # Счетоводство
    'Invoice': 'Фактура',
    'Bill': 'Входяща фактура',
    'Payment': 'Плащане',
    'Journal': 'Дневник',
    'Account': 'Сметка',
    'Tax': 'Данък',
    'Debit': 'Дебит',
    'Credit': 'Кредит',
    'Balance': 'Салдо',
    'Due Date': 'Краен срок',
    'Amount': 'Сума',
    'Total': 'Общо',
    'Subtotal': 'Междинна сума',
    'Untaxed Amount': 'Сума без ДДС',
    'Tax Amount': 'Сума на данъка',
    
    # Склад
    'Warehouse': 'Склад',
    'Location': 'Местоположение',
    'Stock': 'Наличност',
    'Quantity': 'Количество',
    'Unit of Measure': 'Мерна единица',
    'Delivery': 'Доставка',
    'Shipment': 'Пратка',
    'Receipt': 'Получаване',
    
    # Продукти
    'Product': 'Продукт',
    'Product Category': 'Категория продукти',
    'Unit Price': 'Единична цена',
    'Cost': 'Себестойност',
    'Barcode': 'Баркод',
    
    # HR
    'Employee': 'Служител',
    'Department': 'Отдел',
    'Job Position': 'Длъжност',
    'Contract': 'Договор',
    'Payslip': 'Фиш за заплата',
    'Leave': 'Отпуска',
    
    # Състояния
    'Draft': 'Чернова',
    'Confirmed': 'Потвърдено',
    'Done': 'Приключено',
    'Cancelled': 'Отменено',
    'Posted': 'Осчетоводено',
    'Paid': 'Платено',
    
    # Бутони
    'Confirm': 'Потвърди',
    'Cancel': 'Отмени',
    'Print': 'Печат',
    'Send by Email': 'Изпрати по имейл',
    'Validate': 'Валидирай',
    'Post': 'Осчетоводи',
    'Register Payment': 'Регистрирай плащане',
}


def create_standard_terminology() -> TranslationTerminology:
    """Създава стандартна терминология"""
    terminology = TranslationTerminology()
    
    for source, translation in STANDARD_ODOO_TERMS_BG.items():
        terminology.add_term(source, translation)
        
    return terminology
