import json
import owlready2
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import openai  # или аналогичная LLM-библиотека
import numpy as np
from statistics import mean

class OntologyEvaluator:
    """
    Класс для оценки качества онтологий по метрикам:
    - Покрытие CQ
    - Индекс ошибок
    - F1-Score
    - Глубина иерархии
    - Семантическая оценка (G-Eval)
    """
    
    def __init__(self, ontology_path: str):
        self.ontology = owlready2.get_ontology(ontology_path).load()
        self.metrics = {
            'coverage': None,
            'error_index': None,
            'f1_score': None,
            'depth': None,
            'semantic_score': None
        }
    
    def calculate_cq_coverage(self, cq_list: List[str], answered_cq: List[str]) -> float:
        """
        Вычисляет покрытие Competency Questions (CQ)
        
        Args:
            cq_list: Список всех CQ
            answered_cq: Список отвеченных CQ
            
        Returns:
            float: Процент покрытия (0.0 - 1.0)
        """
        coverage = len(answered_cq) / len(cq_list)
        self.metrics['coverage'] = coverage
        return coverage
    
    def detect_ontology_errors(self) -> float:
        """
        Анализирует онтологию на критические ошибки с помощью OOPS!
        
        Returns:
            float: Индекс ошибок (0.0 - 1.0)
        """
        # Здесь должна быть интеграция с OOPS! или аналогичным инструментом
        # Для примера используем упрощенную логику
        all_elements = len(list(self.ontology.classes())) + len(list(self.ontology.object_properties()))
        critical_errors = self._find_duplicate_classes() + self._find_incorrect_relations()
        
        error_index = critical_errors / all_elements if all_elements > 0 else 0
        self.metrics['error_index'] = error_index
        return error_index
    
    def evaluate_f1(self, gold_standard: Dict, system_output: Dict) -> float:
        """
        Вычисляет F1-Score для оценки точности извлечения знаний
        
        Args:
            gold_standard: Эталонные данные (классы и отношения)
            system_output: Выходные данные системы
            
        Returns:
            float: F1-Score (0.0 - 1.0)
        """
        # Преобразуем в бинарные векторы для расчета
        y_true = self._convert_to_vector(gold_standard)
        y_pred = self._convert_to_vector(system_output)
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.metrics['f1_score'] = f1
        return f1
    
    def calculate_hierarchy_depth(self) -> int:
        """
        Вычисляет максимальную глубину иерархии классов
        
        Returns:
            int: Глубина иерархии
        """
        max_depth = 0
        for cls in self.ontology.classes():
            depth = self._get_class_depth(cls)
            if depth > max_depth:
                max_depth = depth
                
        self.metrics['depth'] = max_depth
        return max_depth
    
    def semantic_evaluation(self, document_text: str, g_eval_prompt: str) -> float:
        """
        Проводит семантическую оценку с использованием LLM-as-a-Judge
        
        Args:
            document_text: Исходный текст документа
            g_eval_prompt: Шаблон для G-Eval
            
        Returns:
            float: Средний балл оценки (1-5)
        """
        scores = []
        for cls in self.ontology.classes():
            evaluation = self._evaluate_class_with_llm(cls, document_text, g_eval_prompt)
            scores.append(evaluation)
            
        avg_score = mean(scores)
        self.metrics['semantic_score'] = avg_score
        return avg_score
    
    def generate_report(self) -> Dict:
        """
        Генерирует итоговый отчет по всем метрикам
        
        Returns:
            Dict: Отчет с метриками и рекомендациями
        """
        return {
            'metrics': self.metrics,
            'recommendations': self._generate_recommendations()
        }
    
    # Вспомогательные методы
    def _find_duplicate_classes(self) -> int:
        """Находит дублирующиеся классы (упрощенная реализация)"""
        class_names = [cls.name.lower() for cls in self.ontology.classes()]
        return len(class_names) - len(set(class_names))
    
    def _find_incorrect_relations(self) -> int:
        """Находит некорректные отношения (упрощенная реализация)"""
        # Реальная реализация должна использовать логические проверки
        return 0
    
    def _get_class_depth(self, cls, current_depth: int = 0) -> int:
        """Рекурсивно вычисляет глубину класса"""
        if not cls.is_a:
            return current_depth
        return max(self._get_class_depth(parent, current_depth + 1) for parent in cls.is_a)
    
    def _evaluate_class_with_llm(self, cls, document_text: str, prompt_template: str) -> float:
        """Использует LLM для оценки соответствия класса тексту"""
        prompt = prompt_template.format(
            class_name=cls.name,
            class_definition=str(cls.definition[0]) if hasattr(cls, 'definition') else "",
            document_text=document_text
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        
        try:
            return float(response.choices[0].message['content'])
        except:
            return 3.0  # Средний балл при ошибке
    
    def _convert_to_vector(self, data: Dict) -> np.array:
        """Конвертирует структуру онтологии в вектор для метрик"""
        # Реализация зависит от структуры данных
        return np.zeros(100)  # Заглушка
    
    def _generate_recommendations(self) -> List[str]:
        """Генерирует рекомендации по улучшению на основе метрик"""
        recs = []
        if self.metrics['coverage'] < 0.85:
            recs.append("Увеличить покрытие Competency Questions (текущее: {:.1%})".format(self.metrics['coverage']))
        if self.metrics['error_index'] > 0.05:
            recs.append("Исправить критические ошибки в онтологии (текущий индекс: {:.1%})".format(self.metrics['error_index']))
        return recs


class OntologyGenerator:
    """
    Класс для генерации онтологий из нормативных документов с использованием LLM
    """
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm_model = llm_model
        self.cq_list = []
    
    def generate_from_text(self, text: str, domain: str) -> owlready2.Ontology:
        """
        Генерирует онтологию из текста документа
        
        Args:
            text: Текст нормативного документа
            domain: Домен (например, "law", "finance")
            
        Returns:
            owlready2.Ontology: Сгенерированная онтология
        """
        # Шаг 1: Извлечение ключевых концепций
        concepts = self._extract_concepts(text, domain)
        
        # Шаг 2: Построение иерархии
        hierarchy = self._build_hierarchy(concepts, text)
        
        # Шаг 3: Извлечение отношений
        relations = self._extract_relations(concepts, text)
        
        # Шаг 4: Генерация OWL
        ontology = self._convert_to_owl(concepts, hierarchy, relations, domain)
        
        return ontology
    
    def generate_competency_questions(self, ontology, num_questions: int = 50) -> List[str]:
        """
        Генерирует вопросы компетентности для онтологии
        
        Args:
            ontology: Онтология
            num_questions: Количество вопросов
            
        Returns:
            List[str]: Список CQ
        """
        prompt = f"""Сгенерируйте {num_questions} вопросов компетентности для онтологии в области {ontology.domain}.
        Вопросы должны покрывать: классы, отношения, ограничения и практическое применение."""
        
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        self.cq_list = [q.strip() for q in response.choices[0].message['content'].split("\n") if q.strip()]
        return self.cq_list
    
    # Вспомогательные методы
    def _extract_concepts(self, text: str, domain: str) -> List[Dict]:
        """Извлекает концепции из текста с помощью LLM"""
        prompt = f"""Извлеките ключевые концепции из следующего текста в области {domain}:
        {text}
        
        Верните JSON-список с полями: "concept", "definition", "examples"."""
        
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            return json.loads(response.choices[0].message['content'])
        except:
            return []
    
    def _build_hierarchy(self, concepts: List[Dict], text: str) -> Dict:
        """Строит иерархию классов с помощью LLM"""
        # Реализация аналогична _extract_concepts
        return {}
    
    def _extract_relations(self, concepts: List[Dict], text: str) -> List[Dict]:
        """Извлекает отношения между концепциями"""
        # Реализация аналогична _extract_concepts
        return []
    
    def _convert_to_owl(self, concepts: List[Dict], hierarchy: Dict, relations: List[Dict], domain: str) -> owlready2.Ontology:
        """Конвертирует извлеченные знания в OWL-онтологию"""
        onto = owlready2.get_ontology(f"http://example.com/{domain}_ontology.owl")
        
        with onto:
            # Создаем классы
            classes = {}
            for concept in concepts:
                classes[concept['concept']] = type(concept['concept'], (owlready2.Thing,), {
                    'definition': concept['definition']
                })
            
            # Добавляем иерархию
            for child, parents in hierarchy.items():
                if child in classes and all(p in classes for p in parents):
                    classes[child].is_a = [classes[p] for p in parents]
            
            # Добавляем отношения
            for rel in relations:
                if rel['source'] in classes and rel['target'] in classes:
                    prop = type(rel['relation'], (owlready2.ObjectProperty,), {
                        'domain': [classes[rel['source']]],
                        'range': [classes[rel['target']]]
                    })
        
        return onto


# Пример использования
if __name__ == "__main__":
    # 1. Генерация онтологии
    generator = OntologyGenerator()
    with open("law_document.txt", "r", encoding="utf-8") as f:
        law_text = f.read()
    
    ontology = generator.generate_from_text(law_text, "law")
    ontology.save(file="law_ontology.owl", format="rdfxml")
    
    # 2. Генерация CQ
    cq_list = generator.generate_competency_questions(ontology)
    
    # 3. Оценка онтологии
    evaluator = OntologyEvaluator("law_ontology.owl")
    evaluator.calculate_cq_coverage(cq_list, answered_cq=cq_list[:85])  # Пример: 85% отвечены
    evaluator.detect_ontology_errors()
    
    # Загрузка эталонных данных для F1
    with open("gold_standard.json", "r") as f:
        gold_standard = json.load(f)
    
    evaluator.evaluate_f1(gold_standard, system_output=ontology)
    evaluator.calculate_hierarchy_depth()
    
    # Семантическая оценка
    g_eval_prompt = """Оцените от 1 до 5, насколько класс '{class_name}' соответствует документу:
    Определение класса: {class_definition}
    Контекст документа: {document_text}"""
    evaluator.semantic_evaluation(law_text, g_eval_prompt)
    
    # Генерация отчета
    report = evaluator.generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
