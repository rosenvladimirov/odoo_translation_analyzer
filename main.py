#!/usr/bin/env python3
"""
Odoo Translation Training Data Pipeline
========================================
Главен скрипт за анализ на Odoo source code и генериране на
training data за translation модел.

Usage:
    python main.py analyze /path/to/odoo [--modules sale,purchase,account]
    python main.py generate /path/to/odoo --output ./training_data
    python main.py translate "Sales Order" --model sale.order --field name
"""

import argparse
import json
import sys
from pathlib import Path

from analyzer import OdooSourceAnalyzer
from semantic_analyzer import SemanticAnalyzer, create_standard_terminology
from training_generator import (
    TrainingDataGenerator, 
    InferencePromptBuilder,
    create_few_shot_examples
)


def cmd_analyze(args):
    """Анализира Odoo source code"""
    print(f"Анализиране на {args.path}...")
    
    modules = args.modules.split(',') if args.modules else None
    
    analyzer = OdooSourceAnalyzer(args.path)
    analyzer.analyze(modules)
    
    # Summary
    summary = analyzer.generate_summary()
    print("\n" + "=" * 50)
    print("ОБОБЩЕНИЕ")
    print("=" * 50)
    print(f"Модели: {summary['total_models']}")
    print(f"Полета: {summary['total_fields']}")
    print(f"Методи: {summary['total_methods']}")
    print(f"Views: {summary['total_views']}")
    print(f"Менюта: {summary['total_menus']}")
    print(f"Преводи в PO: {summary['total_translations']}")
    
    if summary['models_by_domain']:
        print("\nМодели по домейн:")
        for domain, count in sorted(summary['models_by_domain'].items(), 
                                    key=lambda x: -x[1]):
            print(f"  {domain}: {count}")
            
    # Export
    output = args.output or 'analysis_output'
    Path(output).mkdir(parents=True, exist_ok=True)
    
    analyzer.export_models(f'{output}/models.json')
    
    # Sample data
    with open(f'{output}/sample_models.json', 'w', encoding='utf-8') as f:
        sample = dict(list(analyzer.models.items())[:5])
        json.dump({k: {
            'name': v.name,
            'description': v.description,
            'module': v.module,
            'fields': list(v.fields.keys())[:10],
            'methods': list(v.methods.keys())[:10]
        } for k, v in sample.items()}, f, ensure_ascii=False, indent=2)
        
    print(f"\nРезултати записани в {output}/")
    

def cmd_generate(args):
    """Генерира training data"""
    print(f"Генериране на training data от {args.path}...")
    
    modules = args.modules.split(',') if args.modules else None
    
    # Analyze
    analyzer = OdooSourceAnalyzer(args.path)
    analyzer.analyze(modules)
    
    # Export models for generator
    models_data = {}
    for name, model in analyzer.models.items():
        models_data[name] = {
            'description': model.description,
            'module': model.module,
            'business_domain': model.business_domain,
            'inherit': model.inherit,
            'fields': {
                fname: {
                    'string': f.string,
                    'help': f.help,
                    'field_type': f.field_type,
                    'selection': f.selection,
                    'comodel_name': f.comodel_name,
                    'required': f.required
                }
                for fname, f in model.fields.items()
            }
        }
        
    # Generate training data
    generator = TrainingDataGenerator(
        models_data,
        analyzer.po_analyzer.translations
    )
    
    generator.generate_from_models()
    generator.filter_samples()
    generator.deduplicate()
    
    # Export
    output = args.output or 'training_output'
    stats = generator.export(output, args.format)
    
    print(f"\nСтатистика:")
    print(f"  Общо samples: {stats['total_samples']}")
    print(f"  По тип:")
    for t, count in stats['by_type'].items():
        print(f"    {t}: {count}")
        

def cmd_semantic(args):
    """Семантичен анализ"""
    print(f"Семантичен анализ на {args.path}...")
    
    modules = args.modules.split(',') if args.modules else None
    
    # Analyze
    analyzer = OdooSourceAnalyzer(args.path)
    analyzer.analyze(modules)
    
    # Convert to dict format
    models_data = {}
    for name, model in analyzer.models.items():
        models_data[name] = {
            'description': model.description,
            'module': model.module,
            'business_domain': model.business_domain,
            'fields': {
                fname: {
                    'string': f.string,
                    'help': f.help,
                    'field_type': f.field_type,
                    'selection': f.selection,
                    'comodel_name': f.comodel_name
                }
                for fname, f in model.fields.items()
            },
            'methods': {
                mname: {
                    'decorators': m.decorators,
                    'docstring': m.docstring,
                    'fields_accessed': m.fields_accessed,
                    'fields_written': m.fields_written
                }
                for mname, m in model.methods.items()
            }
        }
        
    # Semantic analysis
    semantic = SemanticAnalyzer(models_data)
    
    # Sample output
    output = args.output or 'semantic_output'
    Path(output).mkdir(parents=True, exist_ok=True)
    
    # Generate semantic contexts for sample fields
    contexts = []
    for model_name, model in list(models_data.items())[:20]:
        for field_name, field_info in list(model.get('fields', {}).items())[:10]:
            ctx = semantic.analyze_field(model_name, field_name, field_info)
            contexts.append({
                'source': ctx.source_string,
                'model': ctx.model_name,
                'field': ctx.field_name,
                'domain': ctx.business_domain,
                'data_type': ctx.data_type,
                'related_terms': ctx.related_terms[:5],
                'prompt': ctx.to_prompt()
            })
            
    with open(f'{output}/semantic_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)
        
    print(f"\nГенерирани {len(contexts)} семантични контекста")
    print(f"Записани в {output}/semantic_contexts.json")
    
    # Show samples
    print("\nПримерни контексти:")
    for ctx in contexts[:3]:
        print(f"\n  Текст: {ctx['source']}")
        print(f"  Модел: {ctx['model']}.{ctx['field']}")
        print(f"  Домейн: {ctx['domain']}")
        print(f"  Тип данни: {ctx['data_type']}")


def cmd_translate(args):
    """Демо превод с генериран prompt"""
    
    builder = InferencePromptBuilder()
    
    prompt = builder.build_prompt(
        source=args.text,
        model_name=args.model,
        field_name=args.field,
        field_type=args.type,
        business_domain=args.domain
    )
    
    print("=" * 50)
    print("ГЕНЕРИРАН PROMPT ЗА ПРЕВОД")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # Few-shot examples
    if args.few_shot:
        print("\nFEW-SHOT ПРИМЕРИ:")
        for ex in create_few_shot_examples():
            print(f"\n  Source: {ex['source']}")
            print(f"  Context: {ex['context']}")
            print(f"  Target: {ex['target']}")


def cmd_terminology(args):
    """Показва/експортира стандартната терминология"""
    
    terminology = create_standard_terminology()
    
    output = args.output or 'terminology.json'
    terminology.export(output)
    
    print(f"Терминология експортирана в {output}")
    
    # Show sample
    print("\nПримерни термини:")
    from semantic_analyzer import STANDARD_ODOO_TERMS_BG
    for source, target in list(STANDARD_ODOO_TERMS_BG.items())[:20]:
        print(f"  {source} → {target}")


def cmd_info(args):
    """Информация за конкретен модел"""
    
    analyzer = OdooSourceAnalyzer(args.path)
    analyzer.analyze()
    
    model_name = args.model
    
    if model_name not in analyzer.models:
        print(f"Модел '{model_name}' не е намерен")
        print(f"Налични модели (първите 20):")
        for name in list(analyzer.models.keys())[:20]:
            print(f"  {name}")
        return
        
    model = analyzer.models[model_name]
    
    print("=" * 50)
    print(f"МОДЕЛ: {model.name}")
    print("=" * 50)
    print(f"Описание: {model.description}")
    print(f"Модул: {model.module}")
    print(f"Файл: {model.file_path}")
    print(f"Наследява: {model.inherit}")
    print(f"Домейн: {model.business_domain}")
    print(f"Свързани модели: {model.related_models}")
    
    if model.workflow_states:
        print(f"\nWorkflow states:")
        for key, label in model.workflow_states:
            print(f"  {key}: {label}")
            
    print(f"\nПолета ({len(model.fields)}):")
    for fname, finfo in list(model.fields.items())[:20]:
        string = finfo.string or fname
        print(f"  {fname} ({finfo.field_type}): {string}")
        
    print(f"\nМетоди ({len(model.methods)}):")
    for mname, minfo in list(model.methods.items())[:20]:
        decorators = ', '.join(minfo.decorators) if minfo.decorators else ''
        print(f"  {mname}  [{decorators}]")


def main():
    parser = argparse.ArgumentParser(
        description='Odoo Translation Training Data Pipeline'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команди')
    
    # analyze
    p_analyze = subparsers.add_parser('analyze', help='Анализирай Odoo source')
    p_analyze.add_argument('path', help='Път до Odoo')
    p_analyze.add_argument('--modules', '-m', help='Модули (comma-separated)')
    p_analyze.add_argument('--output', '-o', help='Output директория')
    
    # generate
    p_generate = subparsers.add_parser('generate', help='Генерирай training data')
    p_generate.add_argument('path', help='Път до Odoo')
    p_generate.add_argument('--modules', '-m', help='Модули (comma-separated)')
    p_generate.add_argument('--output', '-o', help='Output директория')
    p_generate.add_argument('--format', '-f', default='all',
                           choices=['all', 'jsonl', 'alpaca', 'chatml', 'llama', 'simple'],
                           help='Формат на изхода')
    
    # semantic
    p_semantic = subparsers.add_parser('semantic', help='Семантичен анализ')
    p_semantic.add_argument('path', help='Път до Odoo')
    p_semantic.add_argument('--modules', '-m', help='Модули (comma-separated)')
    p_semantic.add_argument('--output', '-o', help='Output директория')
    
    # translate
    p_translate = subparsers.add_parser('translate', help='Демо превод')
    p_translate.add_argument('text', help='Текст за превод')
    p_translate.add_argument('--model', help='Odoo модел')
    p_translate.add_argument('--field', help='Име на поле')
    p_translate.add_argument('--type', help='Тип на полето')
    p_translate.add_argument('--domain', help='Бизнес домейн')
    p_translate.add_argument('--few-shot', action='store_true', help='Покажи few-shot примери')
    
    # terminology
    p_terminology = subparsers.add_parser('terminology', help='Експортирай терминология')
    p_terminology.add_argument('--output', '-o', help='Output файл')
    
    # info
    p_info = subparsers.add_parser('info', help='Информация за модел')
    p_info.add_argument('path', help='Път до Odoo')
    p_info.add_argument('--model', '-m', required=True, help='Име на модел')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'semantic':
        cmd_semantic(args)
    elif args.command == 'translate':
        cmd_translate(args)
    elif args.command == 'terminology':
        cmd_terminology(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()
        

if __name__ == '__main__':
    main()
