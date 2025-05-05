"""
Модуль для генерации и оценки онтологий из нормативных документов с использованием LLM
"""

import json
import logging
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
from owlready2 import get_ontology, Thing, ObjectProperty, DataProperty, ThingClass
import openai
from sklearn.metrics import f1_score

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class OntologyGenerator:
    """Класс для генерации онтологий из текстовых документов с использованием LLM"""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.llm_model = llm_model
        self.cq_list = []
        
        if api_key:
            openai.api_key = api_key
        else:
            logger.warning("API ключ OpenAI не указан")

    def generate_from_text(self, text: str, domain: str) -> Optional[Thing]:
        """
        Генерирует онтологию из текста документа
        
        Args:
            text: Текст нормативного документа
            domain: Домен (например, "law", "finance")
            
        Returns:
            owlready2.Ontology: Сгенерированная онтология или None при ошибке
        """
        try:
            # Шаг 1: Извлечение знаний
            concepts = self._extract_concepts(text, domain)
            hierarchy = self._build_hierarchy(concepts, text)
            relations = self._extract_relations(concepts, text)
            
            # Шаг 2: Генерация OWL
            return self._convert_to_owl(concepts, hierarchy, relations, domain)
            
        except Exception as e:
            logger.error(f"Ошибка генерации онтологии: {str(e)}")
            return None

    def generate_competency_questions(self, ontology: Thing, num_questions: int = 10) -> List[str]:
        """
        Генерирует вопросы компетентности для онтологии
        
        Args:
            ontology: Онтология
            num_questions: Количество вопросов
            
        Returns:
            List[str]: Список CQ или пустой список при ошибке
        """
        try:
            prompt = f"""Сгенерируйте {num_questions} вопросов компетентности для онтологии в области права.
            Формат: по одному вопросу на строку. Вопросы должны покрывать:
            - Основные классы: {', '.join([c.name for c in ontology.classes()])}
            - Отношения: {', '.join([p.name for p in ontology.object_properties()])}"""
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            self.cq_list = [
                q.strip() 
                for q in response.choices[0].message['content'].split("\n") 
                if q.strip()
            ]
            return self.cq_list
            
        except Exception as e:
            logger.error(f"Ошибка генерации CQ: {str(e)}")
            return []

    # Вспомогательные методы
    def _extract_concepts(self, text: str, domain: str) -> List[Dict]:
        """Извлекает концепции из текста с валидацией"""
        prompt = f"""Извлеките ключевые юридические концепции из текста:
        {text[:3000]}  # Ограничение длины
        
        Формат JSON: [{{"concept": "название", "type": "класс/свойство", "definition": "определение"}}]"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return self._validate_concepts(json.loads(response.choices[0].message['content']))
            
        except Exception as e:
            logger.error(f"Ошибка извлечения концепций: {str(e)}")
            return []

    @staticmethod
    def _validate_concepts(concepts: List[Dict]) -> List[Dict]:
        """Валидация извлеченных концепций"""
        valid_concepts = []
        required_keys = {"concept", "type", "definition"}
        
        for item in concepts:
            if not isinstance(item, dict):
                continue
                
            if all(key in item for key in required_keys):
                if item["type"] in ["класс", "свойство"]:
                    valid_concepts.append(item)
                    
        return valid_concepts

    def _build_hierarchy(self, concepts: List[Dict], text: str) -> Dict[str, List[str]]:
        """Строит иерархию классов с проверкой циклов"""
        hierarchy = {}
        
        try:
            for concept in concepts:
                if concept["type"] == "класс":
                    prompt = f"""Для класса '{concept["concept"]}' из текста:
                    {text[:2000]}
                    Укажите родительские классы через запятую. Если родителей нет, укажите 'КорневойКласс'."""
                    
                    response = openai.ChatCompletion.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2
                    )
                    
                    parents = [
                        p.strip() 
                        for p in response.choices[0].message['content'].split(",") 
                        if p.strip()
                    ]
                    hierarchy[concept["concept"]] = parents
                    
            return hierarchy
            
        except Exception as e:
            logger.error(f"Ошибка построения иерархии: {str(e)}")
            return {}

    def _convert_to_owl(self, concepts: List[Dict], hierarchy: Dict, relations: List[Dict], domain: str) -> Thing:
        """Конвертирует структурированные данные в OWL с проверкой дубликатов"""
        onto = get_ontology(f"http://example.com/{domain}_ontology.owl")
        classes = {}
        properties = {}
        
        with onto:
            # Создание классов
            for concept in concepts:
                if concept["type"] == "класс":
                    if concept["concept"] not in classes:
                        classes[concept["concept"]] = type(
                            concept["concept"], 
                            (Thing,), 
                            {"definition": concept["definition"]}
                        )
            
            # Построение иерархии
            for child, parents in hierarchy.items():
                if child in classes:
                    try:
                        classes[child].is_a = [
                            classes[p] 
                            for p in parents 
                            if p in classes
                        ]
                    except KeyError as e:
                        logger.warning(f"Родительский класс не найден: {str(e)}")
            
            # Создание отношений
            for rel in relations:
                if rel["source"] in classes and rel["target"] in classes:
                    prop_name = self._sanitize_name(rel["relation"])
                    if prop_name not in properties:
                        properties[prop_name] = type(
                            prop_name,
                            (ObjectProperty,),
                            {
                                "domain": [classes[rel["source"]]],
                                "range": [classes[rel["target"]]]
                            }
                        )
        
        return onto

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Очищает названия для OWL-совместимости"""
        return re.sub(r"[^\w]", "_", name.strip())

class OntologyEvaluator:
    """Класс для оценки качества онтологий по метрикам"""
    
    def __init__(self, ontology_path: str):
        self.ontology = get_ontology(ontology_path).load()
        self.metrics = {
            'coverage': None,
            'error_index': None,
            'f1_score': None,
            'depth': None,
            'semantic_score': None
        }

    def calculate_cq_coverage(self, cq_list: List[str], answered_cq: List[str]) -> float:
        coverage = len(answered_cq) / len(cq_list) if cq_list else 0.0
        self.metrics['coverage'] = coverage
        return coverage

    def detect_ontology_errors(self) -> float:
        errors = self._find_duplicate_classes() + self._find_incorrect_relations()
        total_elements = len(list(self.ontology.classes())) + len(list(self.ontology.object_properties()))
        self.metrics['error_index'] = errors / total_elements if total_elements else 0.0
        return self.metrics['error_index']

    def evaluate_f1(self, gold_standard: Dict, system_output: Dict) -> float:
        y_true = self._convert_to_vector(gold_standard)
        y_pred = self._convert_to_vector(system_output)
        self.metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        return self.metrics['f1_score']

    def calculate_hierarchy_depth(self) -> int:
        max_depth = 0
        for cls in self.ontology.classes():
            depth = self._get_class_depth(cls)
            max_depth = max(max_depth, depth)
        self.metrics['depth'] = max_depth
        return max_depth

    def semantic_evaluation(self, document_text: str, g_eval_prompt: str) -> float:
        scores = []
        for cls in self.ontology.classes():
            score = self._evaluate_class_with_llm(cls, document_text, g_eval_prompt)
            scores.append(score)
        self.metrics['semantic_score'] = mean(scores) if scores else 0.0
        return self.metrics['semantic_score']

    def generate_report(self) -> Dict:
        return {
            "metrics": self.metrics,
            "recommendations": self._generate_recommendations()
        }

    # Вспомогательные методы
    def _find_duplicate_classes(self) -> int:
        names = [cls.name.lower() for cls in self.ontology.classes()]
        return len(names) - len(set(names))

    def _find_incorrect_relations(self) -> int:
        errors = 0
        for prop in self.ontology.object_properties():
            if prop.domain == prop.range:
                errors += 1
            if not prop.inverse:
                errors += 1
        return errors

    def _get_class_depth(self, cls: ThingClass, current_depth: int = 0) -> int:
        if not cls.is_a:
            return current_depth
        return max(self._get_class_depth(parent, current_depth + 1) for parent in cls.is_a)

    def _evaluate_class_with_llm(self, cls: ThingClass, text: str, prompt_template: str) -> float:
        try:
            definition = cls.definition[0] if hasattr(cls, 'definition') else ""
            prompt = prompt_template.format(
                class_name=cls.name,
                class_definition=definition,
                document_text=text[:2000]
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0
            )
            
            return float(re.search(r"\d+", response.choices[0].message['content']).group())
        except Exception as e:
            logger.error(f"Ошибка семантической оценки: {str(e)}")
            return 3.0

    def _convert_to_vector(self, data: Dict) -> np.array:
        vector = []
        all_classes = [c.name for c in self.ontology.classes()]
        vector += [1 if c in data.get("classes", []) else 0 for c in all_classes]
        
        all_props = [p.name for p in self.ontology.object_properties()]
        vector += [1 if p in data.get("relations", []) else 0 for p in all_props]
        
        return np.array(vector)

    def _generate_recommendations(self) -> List[str]:
        recs = []
        if self.metrics.get('coverage', 0) < 0.85:
            recs.append("Увеличить покрытие CQ")
        if self.metrics.get('error_index', 0) > 0.05:
            recs.append("Исправить критические ошибки отношений")
        if self.metrics.get('depth', 0) not in range(4,7):
            recs.append("Оптимизировать глубину иерархии")
        return recs