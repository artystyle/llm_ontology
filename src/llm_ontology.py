"""
Модуль для генерации и оценки онтологий из нормативных документов с использованием LLM

Содержит два основных класса:
1. OntologyGenerator - для создания онтологий из текстов
2. OntologyEvaluator - для оценки качества онтологий
"""

import json
import logging
import re
from typing import List, Dict, Optional, Union
import numpy as np
from owlready2 import get_ontology, Thing, ObjectProperty, ThingClass
import openai
from sklearn.metrics import f1_score
from statistics import mean

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class OntologyGenerator:
    """Класс для генерации онтологий из текстовых документов с использованием LLM"""
    
    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        Инициализация генератора онтологий.
        
        Args:
            llm_model (str): Название модели LLM (по умолчанию "gpt-3.5-turbo")
            api_key (Optional[str]): API ключ OpenAI
            base_url (Optional[str]): Базовый URL для API (для локальных развертываний)
            organization (Optional[str]): ID организации OpenAI
        """
        self.llm_model = llm_model
        self.cq_list = []
        
        # Настройка клиента OpenAI
        self._setup_openai_client(api_key, base_url, organization)

    def _setup_openai_client(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        organization: Optional[str]
    ) -> None:
        """
        Конфигурирует клиент OpenAI с указанными параметрами.
        
        Args:
            api_key: API ключ OpenAI
            base_url: Базовый URL для API
            organization: ID организации OpenAI
        """
        if api_key:
            openai.api_key = api_key
        else:
            logger.warning("API ключ OpenAI не указан")
        
        if base_url:
            openai.api_base = base_url
            logger.info(f"Используется кастомный BASE_URL: {base_url}")
        
        if organization:
            openai.organization = organization
            logger.info(f"Используется организация: {organization}")

    def generate_from_text(
        self,
        text: str,
        domain: str,
        max_text_length: int = 3000
    ) -> Optional[Thing]:
        """
        Генерирует онтологию из текста документа.
        
        Args:
            text: Текст нормативного документа
            domain: Домен (например, "law", "finance")
            max_text_length: Максимальная длина текста для обработки
            
        Returns:
            Optional[Thing]: Сгенерированная онтология или None при ошибке
        """
        try:
            # Шаг 1: Извлечение знаний
            concepts = self._extract_concepts(text[:max_text_length], domain)
            if not concepts:
                raise ValueError("Не удалось извлечь концепции из текста")
                
            hierarchy = self._build_hierarchy(concepts, text[:max_text_length])
            relations = self._extract_relations(concepts, text[:max_text_length])
            
            # Шаг 2: Генерация OWL
            ontology = self._convert_to_owl(concepts, hierarchy, relations, domain)
            
            logger.info(f"Успешно сгенерирована онтология с {len(list(ontology.classes()))} классами")
            return ontology
            
        except Exception as e:
            logger.error(f"Ошибка генерации онтологии: {str(e)}", exc_info=True)
            return None

    def generate_competency_questions(
        self,
        ontology: Thing,
        num_questions: int = 10,
        temperature: float = 0.5
    ) -> List[str]:
        """
        Генерирует вопросы компетентности (CQ) для онтологии.
        
        Args:
            ontology: Сгенерированная онтология
            num_questions: Количество генерируемых вопросов
            temperature: Параметр температуры для генерации
            
        Returns:
            List[str]: Список сгенерированных вопросов
        """
        try:
            if not ontology.classes():
                raise ValueError("Онтология не содержит классов")
                
            prompt = self._prepare_cq_prompt(ontology, num_questions)
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
            self.cq_list = self._parse_cq_response(response)
            return self.cq_list
            
        except Exception as e:
            logger.error(f"Ошибка генерации CQ: {str(e)}", exc_info=True)
            return []

    def _prepare_cq_prompt(self, ontology: Thing, num_questions: int) -> str:
        """
        Подготавливает промт для генерации вопросов компетентности.
        
        Args:
            ontology: Онтология
            num_questions: Количество вопросов
            
        Returns:
            str: Сформированный промт
        """
        class_list = ', '.join([c.name for c in ontology.classes()])
        prop_list = ', '.join([p.name for p in ontology.object_properties()])
        
        return f"""Сгенерируйте {num_questions} конкретных вопросов компетентности для оценки онтологии.
Классы: {class_list}
Отношения: {prop_list}
Формат: по одному вопросу на строку, без нумерации."""

    def _parse_cq_response(self, response: Dict) -> List[str]:
        """
        Парсит ответ LLM на вопросы компетентности.
        
        Args:
            response: Ответ от API OpenAI
            
        Returns:
            List[str]: Список вопросов
        """
        return [
            q.strip()
            for q in response.choices[0].message['content'].split("\n")
            if q.strip() and not q.strip().startswith(("1.", "2.", "3."))
        ]

    def _extract_concepts(
        self,
        text: str,
        domain: str,
        temperature: float = 0.3
    ) -> List[Dict]:
        """
        Извлекает концепции из текста с помощью LLM.
        
        Args:
            text: Текст для анализа
            domain: Домен онтологии
            temperature: Параметр температуры для генерации
            
        Returns:
            List[Dict]: Список извлеченных концепций
        """
        try:
            prompt = f"""Извлеките ключевые концепции из текста в области {domain}.
Текст: {text[:2000]}
Формат JSON: [{{"concept": "название", "type": "класс/свойство", "definition": "определение"}}]"""
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            
            return self._validate_concepts(json.loads(response.choices[0].message['content']))
            
        except Exception as e:
            logger.error(f"Ошибка извлечения концепций: {str(e)}", exc_info=True)
            return []

    @staticmethod
    def _validate_concepts(data: Union[Dict, List]) -> List[Dict]:
        """
        Валидирует извлеченные концепции.
        
        Args:
            data: Данные концепций из LLM
            
        Returns:
            List[Dict]: Валидные концепции
        """
        valid_concepts = []
        required_keys = {"concept", "type", "definition"}
        
        # Обработка разных форматов ответа
        if isinstance(data, dict) and "concepts" in data:
            items = data["concepts"]
        elif isinstance(data, list):
            items = data
        else:
            return []
        
        for item in items:
            if isinstance(item, dict) and all(k in item for k in required_keys):
                if item["type"].lower() in ["класс", "свойство"]:
                    valid_concepts.append({
                        "concept": item["concept"].strip(),
                        "type": item["type"].lower(),
                        "definition": item["definition"].strip()
                    })
                    
        return valid_concepts

    def _build_hierarchy(
        self,
        concepts: List[Dict],
        text: str,
        temperature: float = 0.2
    ) -> Dict[str, List[str]]:
        """
        Строит иерархию классов с помощью LLM.
        
        Args:
            concepts: Список концепций
            text: Исходный текст
            temperature: Параметр температуры для генерации
            
        Returns:
            Dict[str, List[str]]: Иерархия классов
        """
        hierarchy = {}
        
        try:
            for concept in concepts:
                if concept["type"] == "класс":
                    prompt = f"""Для класса '{concept["concept"]}' из текста:
{text[:1500]}
Укажите родительские классы через запятую. Если родителей нет, укажите 'Thing'."""
                    
                    response = openai.ChatCompletion.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    
                    parents = [
                        p.strip()
                        for p in response.choices[0].message['content'].split(",")
                        if p.strip() and p.strip().lower() != "none"
                    ]
                    hierarchy[concept["concept"]] = parents
                    
            return hierarchy
            
        except Exception as e:
            logger.error(f"Ошибка построения иерархии: {str(e)}", exc_info=True)
            return {}

    def _extract_relations(
        self,
        concepts: List[Dict],
        text: str,
        temperature: float = 0.3
    ) -> List[Dict]:
        """
        Извлекает отношения между концепциями.
        
        Args:
            concepts: Список концепций
            text: Исходный текст
            temperature: Параметр температуры для генерации
            
        Returns:
            List[Dict]: Список отношений
        """
        relations = []
        
        try:
            concept_names = [c["concept"] for c in concepts if c["type"] == "класс"]
            if len(concept_names) < 2:
                return []
                
            prompt = f"""Из текста:
{text[:2000]}
Извлеките отношения между следующими концепциями: {', '.join(concept_names)}
Формат JSON: [{{"source": "класс1", "target": "класс2", "relation": "тип_отношения"}}]"""
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            
            return self._validate_relations(
                json.loads(response.choices[0].message['content']),
                concept_names
            )
            
        except Exception as e:
            logger.error(f"Ошибка извлечения отношений: {str(e)}", exc_info=True)
            return []

    @staticmethod
    def _validate_relations(data: Union[Dict, List], valid_concepts: List[str]) -> List[Dict]:
        """
        Валидирует извлеченные отношения.
        
        Args:
            data: Данные отношений из LLM
            valid_concepts: Список допустимых концепций
            
        Returns:
            List[Dict]: Валидные отношения
        """
        valid_relations = []
        required_keys = {"source", "target", "relation"}
        
        # Обработка разных форматов ответа
        if isinstance(data, dict) and "relations" in data:
            items = data["relations"]
        elif isinstance(data, list):
            items = data
        else:
            return []
        
        for item in items:
            if (isinstance(item, dict) and 
                all(k in item for k in required_keys) and
                item["source"] in valid_concepts and
                item["target"] in valid_concepts):
                
                valid_relations.append({
                    "source": item["source"].strip(),
                    "target": item["target"].strip(),
                    "relation": item["relation"].strip()
                })
                
        return valid_relations

    def _convert_to_owl(
        self,
        concepts: List[Dict],
        hierarchy: Dict[str, List[str]],
        relations: List[Dict],
        domain: str
    ) -> Thing:
        """
        Конвертирует структурированные данные в OWL-онтологию.
        
        Args:
            concepts: Список концепций
            hierarchy: Иерархия классов
            relations: Список отношений
            domain: Домен онтологии
            
        Returns:
            Thing: Созданная онтология
        """
        try:
            onto = get_ontology(f"http://example.com/{domain}.owl")
            classes = {}
            properties = {}
            
            with onto:
                # Создание классов
                for concept in concepts:
                    if concept["type"] == "класс":
                        cls_name = self._sanitize_name(concept["concept"])
                        if cls_name not in classes:
                            classes[cls_name] = type(
                                cls_name,
                                (Thing,),
                                {"definition": [concept["definition"]]}
                            )
                
                # Построение иерархии
                for child, parents in hierarchy.items():
                    child_cls = classes.get(self._sanitize_name(child))
                    if child_cls:
                        parent_classes = [
                            classes[self._sanitize_name(p)] 
                            for p in parents 
                            if self._sanitize_name(p) in classes
                        ]
                        if parent_classes:
                            child_cls.is_a = parent_classes
                
                # Создание отношений
                for rel in relations:
                    src = self._sanitize_name(rel["source"])
                    tgt = self._sanitize_name(rel["target"])
                    if src in classes and tgt in classes:
                        prop_name = self._sanitize_name(rel["relation"])
                        if prop_name not in properties:
                            properties[prop_name] = type(
                                prop_name,
                                (ObjectProperty,),
                                {
                                    "domain": [classes[src]],
                                    "range": [classes[tgt]]
                                }
                            )
            
            return onto
            
        except Exception as e:
            logger.error(f"Ошибка конвертации в OWL: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Нормализует имена для OWL-совместимости.
        
        Args:
            name: Исходное название
            
        Returns:
            str: Нормализованное название
        """
        # Удаление спецсимволов и замена пробелов
        sanitized = re.sub(r"[^\w]", "_", name.strip())
        # Удаление ведущих цифр
        sanitized = re.sub(r"^\d+", "", sanitized)
        return sanitized or "unnamed"

class OntologyEvaluator:
    """Класс для оценки качества онтологий по различным метрикам"""
    
    def __init__(
        self,
        ontology_path: str,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Инициализация оценщика онтологий.
        
        Args:
            ontology_path: Путь к OWL-файлу онтологии
            llm_model: Модель для семантической оценки
            api_key: API ключ OpenAI
            base_url: Базовый URL API
        """
        self.ontology = get_ontology(ontology_path).load()
        self.llm_model = llm_model
        self.metrics = {
            'coverage': None,
            'error_index': None,
            'f1_score': None,
            'depth': None,
            'semantic_score': None
        }
        
        # Настройка клиента OpenAI
        if api_key:
            openai.api_key = api_key
        if base_url:
            openai.api_base = base_url

    def calculate_cq_coverage(
        self,
        cq_list: List[str],
        answered_cq: List[str]
    ) -> float:
        """
        Вычисляет покрытие вопросов компетентности (CQ).
        
        Args:
            cq_list: Полный список CQ
            answered_cq: Список отвеченных CQ
            
        Returns:
            float: Процент покрытия (0.0-1.0)
        """
        if not cq_list:
            logger.warning("Список CQ пуст")
            return 0.0
            
        coverage = len(answered_cq) / len(cq_list)
        self.metrics['coverage'] = coverage
        return coverage

    def detect_ontology_errors(self) -> float:
        """
        Вычисляет индекс ошибок в онтологии.
        
        Returns:
            float: Индекс ошибок (0.0-1.0)
        """
        errors = self._find_duplicate_classes() + self._find_incorrect_relations()
        total_elements = len(list(self.ontology.classes())) + len(list(self.ontology.object_properties()))
        
        if total_elements == 0:
            logger.warning("Онтология не содержит элементов")
            return 1.0
            
        error_index = errors / total_elements
        self.metrics['error_index'] = error_index
        return error_index

    def evaluate_f1(
        self,
        gold_standard: Dict,
        system_output: Dict
    ) -> float:
        """
        Вычисляет F1-score для оценки точности онтологии.
        
        Args:
            gold_standard: Эталонные данные
            system_output: Данные системы
            
        Returns:
            float: F1-score (0.0-1.0)
        """
        try:
            y_true = self._convert_to_vector(gold_standard)
            y_pred = self._convert_to_vector(system_output)
            
            if len(y_true) != len(y_pred):
                logger.error("Размеры векторов не совпадают")
                return 0.0
                
            f1 = f1_score(y_true, y_pred, average='weighted')
            self.metrics['f1_score'] = f1
            return f1
            
        except Exception as e:
            logger.error(f"Ошибка вычисления F1: {str(e)}", exc_info=True)
            return 0.0

    def calculate_hierarchy_depth(self) -> int:
        """
        Вычисляет максимальную глубину иерархии классов.
        
        Returns:
            int: Глубина иерархии
        """
        max_depth = 0
        for cls in self.ontology.classes():
            depth = self._get_class_depth(cls)
            max_depth = max(max_depth, depth)
            
        self.metrics['depth'] = max_depth
        return max_depth

    def semantic_evaluation(
        self,
        document_text: str,
        g_eval_prompt: str,
        temperature: float = 0.0
    ) -> float:
        """
        Проводит семантическую оценку с использованием LLM-as-a-Judge.
        
        Args:
            document_text: Исходный текст документа
            g_eval_prompt: Шаблон для оценки
            temperature: Параметр температуры для генерации
            
        Returns:
            float: Средний балл оценки (1-5)
        """
        scores = []
        
        for cls in self.ontology.classes():
            try:
                definition = cls.definition[0] if hasattr(cls, 'definition') else ""
                prompt = g_eval_prompt.format(
                    class_name=cls.name,
                    class_definition=definition,
                    document_text=document_text[:2000]
                )
                
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                
                # Извлечение числовой оценки из ответа
                match = re.search(r"\b\d\b", response.choices[0].message['content'])
                if match:
                    score = float(match.group())
                    scores.append(max(1, min(5, score)))  # Ограничение 1-5
                else:
                    scores.append(3.0)  # Среднее значение при ошибке
                    
            except Exception as e:
                logger.error(f"Ошибка оценки класса {cls.name}: {str(e)}")
                scores.append(3.0)
                
        avg_score = mean(scores) if scores else 0.0
        self.metrics['semantic_score'] = avg_score
        return avg_score

    def generate_report(self) -> Dict:
        """
        Генерирует отчет по оценке онтологии.
        
        Returns:
            Dict: Отчет с метриками и рекомендациями
        """
        return {
            "metrics": self.metrics,
            "recommendations": self._generate_recommendations(),
            "summary": self._generate_summary()
        }

    def _generate_recommendations(self) -> List[str]:
        """Генерирует рекомендации по улучшению онтологии."""
        recs = []
        
        if self.metrics.get('coverage', 0) < 0.85:
            recs.append("Увеличить покрытие вопросов компетентности (цель ≥85%)")
            
        if self.metrics.get('error_index', 1) > 0.05:
            recs.append("Исправить ошибки в онтологии (цель <5%)")
            
        depth = self.metrics.get('depth', 0)
        if depth < 4 or depth > 6:
            recs.append(f"Оптимизировать глубину иерархии (текущая: {depth}, цель 4-6)")
            
        if self.metrics.get('semantic_score', 0) < 4.0:
            recs.append("Улучшить семантическое соответствие (цель ≥4.0)")
            
        return recs or ["Онтология соответствует всем ключевым метрикам"]

    def _generate_summary(self) -> str:
        """Генерирует текстовое резюме оценки."""
        metrics = self.metrics
        return (
            f"Оценка онтологии:\n"
            f"- Покрытие CQ: {metrics.get('coverage', 0):.1%}\n"
            f"- Индекс ошибок: {metrics.get('error_index', 0):.1%}\n"
            f"- F1-score: {metrics.get('f1_score', 0):.2f}\n"
            f"- Глубина иерархии: {metrics.get('depth', 0)}\n"
            f"- Семантическая оценка: {metrics.get('semantic_score', 0):.1f}/5.0"
        )

    def _find_duplicate_classes(self) -> int:
        """Находит дублирующиеся классы в онтологии."""
        names = [cls.name.lower() for cls in self.ontology.classes()]
        return len(names) - len(set(names))

    def _find_incorrect_relations(self) -> int:
        """Находит некорректные отношения в онтологии."""
        errors = 0
        
        for prop in self.ontology.object_properties():
            # Проверка циклических отношений
            if prop.domain and prop.range and prop.domain[0] == prop.range[0]:
                errors += 1
                
            # Проверка отношений без обратных
            if not prop.inverse_property:
                errors += 1
                
        return errors

    def _get_class_depth(self, cls: ThingClass, current_depth: int = 0) -> int:
        """
        Рекурсивно вычисляет глубину класса в иерархии.
        
        Args:
            cls: Класс для оценки
            current_depth: Текущая глубина
            
        Returns:
            int: Максимальная глубина
        """
        if not cls.is_a:
            return current_depth
            
        return max(
            self._get_class_depth(parent, current_depth + 1)
            for parent in cls.is_a
            if isinstance(parent, ThingClass)
        )

    def _convert_to_vector(self, data: Dict) -> np.array:
        """
        Конвертирует данные онтологии в вектор для метрик.
        
        Args:
            data: Данные для конвертации
            
        Returns:
            np.array: Векторизованные данные
        """
        all_elements = (
            [c.name for c in self.ontology.classes()] +
            [p.name for p in self.ontology.object_properties()]
        )
        
        return np.array([
            1 if elem in data.get('elements', []) else 0
            for elem in all_elements
        ])

    def visualize_hierarchy(self) -> None:
        """Визуализирует иерархию классов онтологии."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Добавление классов и отношений
            for cls in self.ontology.classes():
                G.add_node(cls.name)
                for parent in cls.is_a:
                    if isinstance(parent, ThingClass):
                        G.add_edge(parent.name, cls.name)
            
            # Визуализация
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(
                G, pos, with_labels=True, node_size=2000,
                node_color="skyblue", font_size=10,
                arrowsize=20, arrowstyle="->"
            )
            plt.title("Иерархия классов онтологии")
            plt.show()
            
        except ImportError:
            logger.warning("Для визуализации установите matplotlib и networkx")
        except Exception as e:
            logger.error(f"Ошибка визуализации: {str(e)}")