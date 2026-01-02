#!/usr/bin/env python3
"""
Training Data Generator
=======================
Генерира training data за fine-tuning на LLM модел за преводи.
Поддържа различни формати: JSONL, Alpaca, ChatML, и др.
"""

import json
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingSample:
    """Един training sample"""
    source: str
    target: str
    context: dict
    
    # Metadata
    model_name: str
    field_name: Optional[str]
    translation_type: str
    business_domain: str
    
    def to_alpaca(self) -> dict:
        """Формат за Alpaca-style fine-tuning"""
        instruction = self._build_instruction()
        
        return {
            "instruction": instruction,
            "input": self.source,
            "output": self.target
        }
        
    def to_chatml(self) -> dict:
        """Формат за ChatML (OpenAI style)"""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt()
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": self.target}
            ]
        }
        
    def to_llama_chat(self) -> dict:
        """Формат за Llama 2/3 chat"""
        return {
            "conversations": [
                {
                    "from": "human",
                    "value": self._build_full_prompt()
                },
                {
                    "from": "gpt",
                    "value": self.target
                }
            ]
        }
        
    def to_simple(self) -> dict:
        """Прост формат с всичко"""
        return {
            "source": self.source,
            "target": self.target,
            "model": self.model_name,
            "field": self.field_name,
            "type": self.translation_type,
            "domain": self.business_domain,
            "context": self.context
        }
        
    def _build_instruction(self) -> str:
        """Генерира instruction за Alpaca формат"""
        parts = ["Преведи следния текст от Odoo ERP на български език."]
        
        if self.business_domain:
            parts.append(f"Домейн: {self.business_domain}.")
            
        if self.translation_type == 'field_label':
            parts.append(f"Това е етикет на поле в модел {self.model_name}.")
        elif self.translation_type == 'help_text':
            parts.append(f"Това е помощен текст за поле в {self.model_name}.")
        elif self.translation_type == 'selection_value':
            parts.append(f"Това е стойност от падащо меню в {self.model_name}.")
        elif self.translation_type == 'error_message':
            parts.append("Това е съобщение за грешка.")
        elif self.translation_type == 'button_label':
            parts.append("Това е етикет на бутон.")
            
        return ' '.join(parts)
        
    def _build_system_prompt(self) -> str:
        """Генерира system prompt за ChatML"""
        return """Ти си експертен преводач за Odoo ERP система. Твоята задача е да превеждаш текстове от английски на български, като спазваш:

1. Използвай установената Odoo терминология на български
2. Съобразявай се с контекста (модел, тип поле, бизнес домейн)
3. Запазвай техническите термини където е уместно
4. Преводът трябва да е естествен и разбираем за български потребители
5. Спазвай стила на Odoo - кратък и ясен

Отговаряй само с превода, без допълнителни обяснения."""

    def _build_user_prompt(self) -> str:
        """Генерира user prompt"""
        parts = [f'Преведи: "{self.source}"']
        
        context_parts = []
        if self.model_name:
            context_parts.append(f"Модел: {self.model_name}")
        if self.field_name:
            context_parts.append(f"Поле: {self.field_name}")
        if self.business_domain:
            context_parts.append(f"Домейн: {self.business_domain}")
        if self.translation_type:
            type_names = {
                'field_label': 'Етикет на поле',
                'help_text': 'Помощен текст',
                'selection_value': 'Стойност от списък',
                'error_message': 'Съобщение за грешка',
                'button_label': 'Етикет на бутон',
                'menu_item': 'Елемент от меню',
                'model_description': 'Описание на модел'
            }
            context_parts.append(f"Тип: {type_names.get(self.translation_type, self.translation_type)}")
            
        if context_parts:
            parts.append("Контекст: " + ", ".join(context_parts))
            
        return '\n'.join(parts)
        
    def _build_full_prompt(self) -> str:
        """Генерира пълен prompt"""
        return f"""{self._build_system_prompt()}

{self._build_user_prompt()}"""


class TrainingDataGenerator:
    """Генератор на training data"""
    
    def __init__(self, models_data: dict, translations: dict = None):
        self.models = models_data
        self.translations = translations or {}
        self.samples = []
        
    def generate_from_models(self):
        """Генерира samples от анализираните модели"""
        
        for model_name, model in self.models.items():
            # Model description
            if model.get('description'):
                translation = self._get_translation(model['description'])
                if translation:
                    self.samples.append(TrainingSample(
                        source=model['description'],
                        target=translation,
                        context={'inherit': model.get('inherit', [])},
                        model_name=model_name,
                        field_name=None,
                        translation_type='model_description',
                        business_domain=model.get('business_domain', '')
                    ))
                    
            # Fields
            for field_name, field_info in model.get('fields', {}).items():
                # Field label
                if field_info.get('string'):
                    translation = self._get_translation(field_info['string'])
                    if translation:
                        self.samples.append(TrainingSample(
                            source=field_info['string'],
                            target=translation,
                            context={
                                'field_type': field_info.get('field_type'),
                                'help': field_info.get('help'),
                                'comodel': field_info.get('comodel_name'),
                                'required': field_info.get('required', False)
                            },
                            model_name=model_name,
                            field_name=field_name,
                            translation_type='field_label',
                            business_domain=model.get('business_domain', '')
                        ))
                        
                # Help text
                if field_info.get('help'):
                    translation = self._get_translation(field_info['help'])
                    if translation:
                        self.samples.append(TrainingSample(
                            source=field_info['help'],
                            target=translation,
                            context={
                                'field_label': field_info.get('string'),
                                'field_type': field_info.get('field_type')
                            },
                            model_name=model_name,
                            field_name=field_name,
                            translation_type='help_text',
                            business_domain=model.get('business_domain', '')
                        ))
                        
                # Selection values
                if field_info.get('selection'):
                    for key, label in field_info['selection']:
                        translation = self._get_translation(label)
                        if translation:
                            self.samples.append(TrainingSample(
                                source=label,
                                target=translation,
                                context={
                                    'selection_key': key,
                                    'field_label': field_info.get('string'),
                                    'all_values': [k for k, v in field_info['selection']]
                                },
                                model_name=model_name,
                                field_name=field_name,
                                translation_type='selection_value',
                                business_domain=model.get('business_domain', '')
                            ))
                            
        return self
        
    def _get_translation(self, source: str) -> Optional[str]:
        """Взима превод от наличните преводи"""
        if source in self.translations:
            translations = self.translations[source]
            # Предпочитай български
            if isinstance(translations, dict):
                return translations.get('bg') or translations.get('bg_BG')
            return translations
        return None
        
    def filter_samples(self, min_source_len: int = 2, max_source_len: int = 500):
        """Филтрира samples по критерии"""
        self.samples = [
            s for s in self.samples
            if min_source_len <= len(s.source) <= max_source_len
            and s.target  # Има превод
            and s.source != s.target  # Различни
        ]
        return self
        
    def deduplicate(self):
        """Премахва дубликати"""
        seen = set()
        unique = []
        
        for sample in self.samples:
            key = (sample.source, sample.target)
            if key not in seen:
                seen.add(key)
                unique.append(sample)
                
        self.samples = unique
        return self
        
    def split(self, train_ratio: float = 0.9, seed: int = 42):
        """Разделя на train/validation"""
        random.seed(seed)
        shuffled = self.samples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        
        return shuffled[:split_idx], shuffled[split_idx:]
        
    def export(self, output_dir: str, format: str = 'all'):
        """Експортира в различни формати"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train, val = self.split()
        
        formats = {
            'jsonl': self._export_jsonl,
            'alpaca': self._export_alpaca,
            'chatml': self._export_chatml,
            'llama': self._export_llama,
            'simple': self._export_simple
        }
        
        if format == 'all':
            for fmt, func in formats.items():
                func(train, val, output_path)
        elif format in formats:
            formats[format](train, val, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        # Statistics
        stats = {
            'total_samples': len(self.samples),
            'train_samples': len(train),
            'val_samples': len(val),
            'unique_sources': len(set(s.source for s in self.samples)),
            'by_type': {}
        }
        
        for sample in self.samples:
            t = sample.translation_type
            stats['by_type'][t] = stats['by_type'].get(t, 0) + 1
            
        with open(output_path / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        print(f"\nЕкспортирани {len(self.samples)} samples в {output_dir}")
        print(f"  Train: {len(train)}, Validation: {len(val)}")
        
        return stats
        
    def _export_jsonl(self, train: list, val: list, output_path: Path):
        """JSONL формат"""
        for name, data in [('train', train), ('val', val)]:
            with open(output_path / f'{name}.jsonl', 'w', encoding='utf-8') as f:
                for sample in data:
                    f.write(json.dumps(sample.to_simple(), ensure_ascii=False) + '\n')
                    
    def _export_alpaca(self, train: list, val: list, output_path: Path):
        """Alpaca формат"""
        for name, data in [('train', train), ('val', val)]:
            samples = [s.to_alpaca() for s in data]
            with open(output_path / f'alpaca_{name}.json', 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
                
    def _export_chatml(self, train: list, val: list, output_path: Path):
        """ChatML формат"""
        for name, data in [('train', train), ('val', val)]:
            with open(output_path / f'chatml_{name}.jsonl', 'w', encoding='utf-8') as f:
                for sample in data:
                    f.write(json.dumps(sample.to_chatml(), ensure_ascii=False) + '\n')
                    
    def _export_llama(self, train: list, val: list, output_path: Path):
        """Llama chat формат"""
        for name, data in [('train', train), ('val', val)]:
            with open(output_path / f'llama_{name}.jsonl', 'w', encoding='utf-8') as f:
                for sample in data:
                    f.write(json.dumps(sample.to_llama_chat(), ensure_ascii=False) + '\n')
                    
    def _export_simple(self, train: list, val: list, output_path: Path):
        """Прост CSV-подобен формат"""
        for name, data in [('train', train), ('val', val)]:
            with open(output_path / f'simple_{name}.tsv', 'w', encoding='utf-8') as f:
                f.write("source\ttarget\tmodel\tfield\ttype\tdomain\n")
                for s in data:
                    f.write(f"{s.source}\t{s.target}\t{s.model_name}\t{s.field_name or ''}\t{s.translation_type}\t{s.business_domain}\n")


class InferencePromptBuilder:
    """Генерира prompts за inference (реално използване на обучения модел)"""
    
    def __init__(self, terminology: dict = None):
        self.terminology = terminology or {}
        
    def build_prompt(self, source: str, model_name: str = None, 
                     field_name: str = None, field_type: str = None,
                     translation_type: str = 'field_label',
                     business_domain: str = None,
                     similar_translations: list = None) -> str:
        """Генерира prompt за превод"""
        
        parts = []
        
        # System instruction
        parts.append("Преведи следния текст от Odoo ERP система на български език.")
        
        # Context
        context_parts = []
        if model_name:
            context_parts.append(f"Модел: {model_name}")
        if field_name:
            context_parts.append(f"Поле: {field_name}")
        if field_type:
            context_parts.append(f"Тип: {field_type}")
        if business_domain:
            context_parts.append(f"Домейн: {business_domain}")
            
        if context_parts:
            parts.append("Контекст: " + ", ".join(context_parts))
            
        # Similar translations for consistency
        if similar_translations:
            parts.append("Подобни преводи за референция:")
            for src, tgt in similar_translations[:3]:
                parts.append(f'  "{src}" → "{tgt}"')
                
        # The text to translate
        parts.append(f'\nТекст за превод: "{source}"')
        parts.append("\nПревод:")
        
        return '\n'.join(parts)
        
    def build_batch_prompt(self, items: list) -> str:
        """Генерира prompt за batch превод"""
        
        parts = ["Преведи следните текстове от Odoo ERP на български:", ""]
        
        for i, item in enumerate(items, 1):
            context = []
            if item.get('model'):
                context.append(f"модел: {item['model']}")
            if item.get('field'):
                context.append(f"поле: {item['field']}")
                
            context_str = f" ({', '.join(context)})" if context else ""
            parts.append(f'{i}. "{item["source"]}"{context_str}')
            
        parts.append("\nПреводи (по един на ред):")
        
        return '\n'.join(parts)


def create_few_shot_examples() -> list:
    """Създава few-shot примери за in-context learning"""
    
    return [
        {
            "source": "Sales Order",
            "context": "Модел: sale.order, Домейн: Продажби",
            "target": "Поръчка за продажба"
        },
        {
            "source": "Expected Date",
            "context": "Модел: purchase.order, Поле: date_planned, Тип: Datetime",
            "target": "Очаквана дата"
        },
        {
            "source": "The quantity to deliver must be positive.",
            "context": "Модел: stock.move, Тип: error_message",
            "target": "Количеството за доставка трябва да е положително число."
        },
        {
            "source": "Confirm",
            "context": "Тип: button_label, Домейн: Продажби",
            "target": "Потвърди"
        },
        {
            "source": "This field is used to compute the remaining amount on the invoice.",
            "context": "Модел: account.move, Поле: amount_residual, Тип: help_text",
            "target": "Това поле се използва за изчисляване на оставащата сума по фактурата."
        }
    ]
