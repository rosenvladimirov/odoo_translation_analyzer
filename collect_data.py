#!/usr/bin/env python3
"""
Odoo Training Data Collector
=============================
Събира training данни от множество Odoo източници:
- Odoo Community (различни версии)
- Odoo Enterprise
- OCA модули
- Съществуващи PO преводи

Генерира максимално количество качествени training samples.
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

# Импортирай нашите анализатори
from analyzer import OdooSourceAnalyzer
from semantic_analyzer import SemanticAnalyzer, STANDARD_ODOO_TERMS_BG
from training_generator import TrainingDataGenerator, TrainingSample


class OdooDataCollector:
    """Събира данни от множество Odoo източници"""
    
    # Източници за сваляне
    SOURCES = {
        # Odoo Community
        'odoo-18': {
            'url': 'https://github.com/odoo/odoo.git',
            'branch': '18.0',
            'type': 'odoo'
        },
        'odoo-17': {
            'url': 'https://github.com/odoo/odoo.git', 
            'branch': '17.0',
            'type': 'odoo'
        },
        'odoo-16': {
            'url': 'https://github.com/odoo/odoo.git',
            'branch': '16.0', 
            'type': 'odoo'
        },
        
        # OCA модули (отлични преводи)
        'oca-l10n-bulgaria': {
            'url': 'https://github.com/OCA/l10n-bulgaria.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-account': {
            'url': 'https://github.com/OCA/account-financial-tools.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-sale': {
            'url': 'https://github.com/OCA/sale-workflow.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-purchase': {
            'url': 'https://github.com/OCA/purchase-workflow.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-stock': {
            'url': 'https://github.com/OCA/stock-logistics-workflow.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-hr': {
            'url': 'https://github.com/OCA/hr.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-manufacture': {
            'url': 'https://github.com/OCA/manufacture.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-project': {
            'url': 'https://github.com/OCA/project.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-crm': {
            'url': 'https://github.com/OCA/crm.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-pos': {
            'url': 'https://github.com/OCA/pos.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-reporting': {
            'url': 'https://github.com/OCA/reporting-engine.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-website': {
            'url': 'https://github.com/OCA/website.git',
            'branch': '16.0',
            'type': 'oca'
        },
        'oca-server-tools': {
            'url': 'https://github.com/OCA/server-tools.git',
            'branch': '16.0',
            'type': 'oca'
        },
    }
    
    def __init__(self, work_dir: str = './odoo_sources'):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_translations = {}  # source -> {lang: translation}
        self.all_models = {}  # model_name -> ModelInfo
        self.all_samples = []
        
    def download_sources(self, sources: list = None):
        """Сваля source code от GitHub"""
        
        sources = sources or list(self.SOURCES.keys())
        
        print("=" * 60)
        print("СВАЛЯНЕ НА ODOO ИЗТОЧНИЦИ")
        print("=" * 60)
        
        for name in sources:
            if name not in self.SOURCES:
                print(f"⚠ Непознат източник: {name}")
                continue
                
            source = self.SOURCES[name]
            target_dir = self.work_dir / name
            
            if target_dir.exists():
                print(f"✓ {name} вече съществува")
                continue
                
            print(f"⬇ Сваляне на {name}...")
            
            try:
                cmd = [
                    'git', 'clone',
                    '--depth', '1',
                    '--branch', source['branch'],
                    source['url'],
                    str(target_dir)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  ✓ {name} свален успешно")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Грешка при сваляне на {name}: {e}")
                
    def analyze_source(self, source_path: Path) -> dict:
        """Анализира един източник"""
        
        analyzer = OdooSourceAnalyzer(str(source_path))
        analyzer.analyze()
        
        return {
            'models': analyzer.models,
            'translations': analyzer.po_analyzer.translations,
            'summary': analyzer.generate_summary()
        }
        
    def collect_all(self, parallel: bool = True):
        """Събира данни от всички източници"""
        
        print("\n" + "=" * 60)
        print("АНАЛИЗ НА ИЗТОЧНИЦИ")
        print("=" * 60)
        
        sources_to_analyze = []
        for name in self.SOURCES.keys():
            source_path = self.work_dir / name
            if source_path.exists():
                sources_to_analyze.append((name, source_path))
                
        if not sources_to_analyze:
            print("Няма свалени източници. Първо изпълни download_sources()")
            return
            
        results = {}
        
        if parallel and len(sources_to_analyze) > 1:
            # Паралелен анализ
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.analyze_source, path): name 
                    for name, path in sources_to_analyze
                }
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                        print(f"✓ {name}: {results[name]['summary']['total_models']} модела, "
                              f"{results[name]['summary']['total_translations']} превода")
                    except Exception as e:
                        print(f"✗ {name}: {e}")
        else:
            # Последователен анализ
            for name, path in sources_to_analyze:
                print(f"Анализиране на {name}...")
                try:
                    results[name] = self.analyze_source(path)
                    print(f"  ✓ {results[name]['summary']['total_models']} модела, "
                          f"{results[name]['summary']['total_translations']} превода")
                except Exception as e:
                    print(f"  ✗ Грешка: {e}")
                    
        # Обедини резултатите
        self._merge_results(results)
        
        return results
        
    def _merge_results(self, results: dict):
        """Обединява резултати от различни източници"""
        
        print("\n" + "=" * 60)
        print("ОБЕДИНЯВАНЕ НА ДАННИ")
        print("=" * 60)
        
        # Обедини преводи (предпочитай по-нови версии)
        for name, data in sorted(results.items(), reverse=True):
            for source, translations in data['translations'].items():
                if source not in self.all_translations:
                    self.all_translations[source] = translations
                else:
                    # Merge, като запазваме съществуващите
                    if isinstance(translations, dict):
                        if isinstance(self.all_translations[source], dict):
                            self.all_translations[source].update(translations)
                        else:
                            self.all_translations[source] = translations
                            
        # Обедини модели
        for name, data in results.items():
            for model_name, model in data['models'].items():
                if model_name not in self.all_models:
                    self.all_models[model_name] = {
                        'name': model.name,
                        'description': model.description,
                        'module': model.module,
                        'business_domain': model.business_domain,
                        'inherit': model.inherit,
                        'fields': {},
                        'sources': []
                    }
                    
                # Добави полета
                for fname, finfo in model.fields.items():
                    self.all_models[model_name]['fields'][fname] = {
                        'string': finfo.string,
                        'help': finfo.help,
                        'field_type': finfo.field_type,
                        'selection': finfo.selection,
                        'comodel_name': finfo.comodel_name,
                        'required': finfo.required
                    }
                    
                self.all_models[model_name]['sources'].append(name)
                
        print(f"Общо модели: {len(self.all_models)}")
        print(f"Общо преводи: {len(self.all_translations)}")
        
    def generate_samples(self):
        """Генерира training samples"""
        
        print("\n" + "=" * 60)
        print("ГЕНЕРИРАНЕ НА TRAINING SAMPLES")
        print("=" * 60)
        
        generator = TrainingDataGenerator(
            self.all_models,
            self.all_translations
        )
        
        generator.generate_from_models()
        
        # Добави стандартната терминология като допълнителни примери
        for source, target in STANDARD_ODOO_TERMS_BG.items():
            if source not in [s.source for s in generator.samples]:
                generator.samples.append(TrainingSample(
                    source=source,
                    target=target,
                    context={'source': 'standard_terminology'},
                    model_name='',
                    field_name=None,
                    translation_type='terminology',
                    business_domain=''
                ))
                
        generator.filter_samples()
        generator.deduplicate()
        
        self.all_samples = generator.samples
        
        print(f"Генерирани samples: {len(self.all_samples)}")
        
        # Статистики по тип
        by_type = defaultdict(int)
        for s in self.all_samples:
            by_type[s.translation_type] += 1
            
        print("\nПо тип:")
        for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")
            
        return generator
        
    def augment_samples(self):
        """Разширява samples с вариации"""
        
        print("\n" + "=" * 60)
        print("AUGMENTATION НА ДАННИ")
        print("=" * 60)
        
        augmented = []
        
        for sample in self.all_samples:
            # Оригинален sample
            augmented.append(sample)
            
            # Вариация 1: Само текст без контекст
            if sample.model_name:
                augmented.append(TrainingSample(
                    source=sample.source,
                    target=sample.target,
                    context={},
                    model_name='',
                    field_name=None,
                    translation_type=sample.translation_type,
                    business_domain=''
                ))
                
            # Вариация 2: С различен контекст формат
            if sample.business_domain:
                augmented.append(TrainingSample(
                    source=sample.source,
                    target=sample.target,
                    context={'domain_only': True},
                    model_name='',
                    field_name=None,
                    translation_type=sample.translation_type,
                    business_domain=sample.business_domain
                ))
                
        # Дедупликация
        seen = set()
        unique = []
        for s in augmented:
            key = hashlib.md5(f"{s.source}|{s.target}|{s.model_name}".encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(s)
                
        self.all_samples = unique
        print(f"Samples след augmentation: {len(self.all_samples)}")
        
    def export(self, output_dir: str = './training_data'):
        """Експортира всички формати"""
        
        print("\n" + "=" * 60)
        print("ЕКСПОРТИРАНЕ")
        print("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Генератор за форматиране
        generator = TrainingDataGenerator({}, {})
        generator.samples = self.all_samples
        
        generator.export(output_dir, format='all')
        
        # Допълнително: raw данни за анализ
        with open(output_path / 'all_translations.json', 'w', encoding='utf-8') as f:
            json.dump(self.all_translations, f, ensure_ascii=False, indent=2)
            
        with open(output_path / 'all_models.json', 'w', encoding='utf-8') as f:
            json.dump(self.all_models, f, ensure_ascii=False, indent=2)
            
        print(f"\nВсичко експортирано в: {output_dir}/")
        
    def print_summary(self):
        """Показва обобщение"""
        
        print("\n" + "=" * 60)
        print("ОБОБЩЕНИЕ")
        print("=" * 60)
        
        print(f"Модели: {len(self.all_models)}")
        print(f"Уникални преводи: {len(self.all_translations)}")
        print(f"Training samples: {len(self.all_samples)}")
        
        # Преводи по език
        langs = defaultdict(int)
        for source, trans in self.all_translations.items():
            if isinstance(trans, dict):
                for lang in trans.keys():
                    langs[lang] += 1
                    
        print("\nПреводи по език:")
        for lang, count in sorted(langs.items(), key=lambda x: -x[1])[:10]:
            print(f"  {lang}: {count}")
            
        # Топ домейни
        domains = defaultdict(int)
        for model in self.all_models.values():
            if model.get('business_domain'):
                domains[model['business_domain']] += 1
                
        print("\nМодели по домейн:")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
            print(f"  {domain}: {count}")


def main():
    """Главна функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Събира Odoo training данни')
    parser.add_argument('--work-dir', '-w', default='./odoo_sources',
                       help='Работна директория за source code')
    parser.add_argument('--output', '-o', default='./training_data',
                       help='Изходна директория за training data')
    parser.add_argument('--download', '-d', action='store_true',
                       help='Свали source code от GitHub')
    parser.add_argument('--sources', '-s', 
                       help='Конкретни източници (comma-separated)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Без augmentation')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Бърз режим - само Odoo 18')
    
    args = parser.parse_args()
    
    collector = OdooDataCollector(args.work_dir)
    
    # Определи източници
    if args.quick:
        sources = ['odoo-18']
    elif args.sources:
        sources = args.sources.split(',')
    else:
        sources = None  # Всички
        
    # Сваляне
    if args.download:
        collector.download_sources(sources)
        
    # Анализ
    collector.collect_all(parallel=True)
    
    # Генериране
    collector.generate_samples()
    
    # Augmentation
    if not args.no_augment:
        collector.augment_samples()
        
    # Експорт
    collector.export(args.output)
    
    # Обобщение
    collector.print_summary()
    
    print("\n✓ Готово!")
    print(f"\nСледваща стъпка - качи {args.output}/ в cloud и стартирай обучение.")


if __name__ == '__main__':
    main()
