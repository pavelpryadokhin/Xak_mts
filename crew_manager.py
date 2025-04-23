import os
import sys
from typing import List, Dict, Any
import json
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Добавляем пути к модулям агентов
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "Intent Agent"))
sys.path.append(os.path.join(current_dir, "Emotion_Agent"))

# Импортируем агентов
from Intent_Agent.inference import IntentPredictor, db_categories
from Emotion_Agent.inference import EmotionPredictor
from Knowledge_Agent.inference import KnowledgeAgent            

class DialogCrew:
    """Класс для управления экипажем агентов обработки диалогов с использованием CrewAI"""
    
    def __init__(self):
        # Инициализируем предикторы
        self.intent_model_path = os.path.join(current_dir, "Intent Agent", "bert_model")
        self.intent_predictor = IntentPredictor(model_path=self.intent_model_path, labels_dict=db_categories)
        self.emotion_predictor = EmotionPredictor()
        self.knowledge_agent = KnowledgeAgent(
            embedding_api_url=os.getenv("EMBEDDING_API_URL"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
            completion_api_url=os.getenv("COMPLETION_API_URL"),
            completion_api_key=os.getenv("COMPLETION_API_KEY"),
            embedding_model=os.getenv("EMBEDDING_MODEL"),
            completion_model=os.getenv("COMPLETION_MODEL"),
            qdrant_host=os.getenv("QDRANT_HOST"),
            qdrant_port=int(os.getenv("QDRANT_PORT"))
        )
        
        # Создаем экипаж агентов CrewAI
        self.crew = self._setup_crew()
    
    def _setup_crew(self) -> Crew:
        """Настраивает экипаж CrewAI с агентами и задачами"""
        
        # 1. Создаем агентов CrewAI
        intent_agent = Agent(
            role="Intent Analyzer",
            goal="Определить намерение пользователя в диалоге",
            backstory="Я анализирую диалоги и определяю категорию запроса пользователя.",
            verbose=True
        )
        
        emotion_agent = Agent(
            role="Emotion Analyzer",
            goal="Определить эмоциональный тон диалога",
            backstory="Я анализирую эмоциональный окрас сообщений пользователя.",
            verbose=True
        )
        
        knowledge_agent = Agent(
            role="Knowledge Provider",
            goal="Предоставить информацию по запросу пользователя",
            backstory="Я поставляю знания и отвечаю на запросы пользователей.",
            verbose=True
        )
        
        action_agent = Agent(
            role="Action Recommender",
            goal="Рекомендовать действия на основе анализа диалога",
            backstory="Я рекомендую оптимальные действия в зависимости от намерения и эмоций пользователя.",
            verbose=True
        )
        
        summary_agent = Agent(
            role="Dialog Summarizer",
            goal="Создать краткое резюме диалога для CRM",
            backstory="Я составляю краткие и информативные резюме диалогов для CRM-системы.",
            verbose=True
        )
        
        quality_agent = Agent(
            role="Quality Assessor",
            goal="Оценить качество обработки диалога",
            backstory="Я оцениваю качество ответов и обработки диалогов.",
            verbose=True
        )
        
        # 2. Создаем задачи
        intent_task = Task(
            description="Проанализировать диалог и определить намерение пользователя",
            expected_output="JSON с категорией диалога и уровнем уверенности",
            agent=intent_agent
        )
        
        emotion_task = Task(
            description="Проанализировать эмоциональный тон диалога",
            expected_output="JSON с определенной эмоцией и уровнем уверенности",
            agent=emotion_agent
        )
        
        knowledge_task = Task(
            description="Предоставить информацию по запросу пользователя, основываясь на его намерении",
            expected_output="JSON с ответом на запрос пользователя",
            agent=knowledge_agent, # Зависит от результата задачи intent_task
        )
        
        action_task = Task(
            description="Рекомендовать действия, основываясь на намерении и эмоциях пользователя",
            expected_output="JSON со списком рекомендуемых действий",
            agent=action_agent,
            context=[intent_task, emotion_task]  # Зависит от результатов задач intent_task и emotion_task
        )
        
        summary_task = Task(
            description="Создать краткое резюме диалога для CRM",
            expected_output="JSON с кратким резюме для CRM",
            agent=summary_agent,
            context=[intent_task, emotion_task, knowledge_task, action_task]
        )
        
        quality_task = Task(
            description="Оценить качество обработки диалога",
            expected_output="JSON с оценкой качества",
            agent=quality_agent,
            context=[intent_task, emotion_task, knowledge_task, action_task, summary_task]
        )
        
        # 3. Создаем экипаж с задачами
        crew = Crew(
            agents=[intent_agent, emotion_agent, knowledge_agent, action_agent, summary_agent, quality_agent],
            tasks=[intent_task, emotion_task, knowledge_task, action_task, summary_task, quality_task],
            verbose=2,
            process=Process.sequential  # Последовательное выполнение задач
        )
        
        return crew
    
    def _extract_user_messages(self, dialog: List[str]) -> str:
        """Извлекает сообщения пользователя из диалога"""
        return " ".join([msg for i, msg in enumerate(dialog) if i % 2 == 0])
    
    def process_dialog_with_actual_models(self, dialog: List[str]) -> Dict[str, Any]:
        """
        Обрабатывает диалог с использованием фактических моделей агентов
        (используется как fallback, когда CrewAI по каким-то причинам не может быть использован)
        """
        user_text = self._extract_user_messages(dialog)
        
        # Получаем результаты от доступных агентов
        intent_result = self.intent_predictor.predict(user_text)
        
        raw_emotion_result = self.emotion_predictor.predict(user_text)
        emotion_result = {
            'label': raw_emotion_result[0]['label'],
            'score': float(raw_emotion_result[0]['score'])
        }
        
        # Заглушки для других агентов (пока не реализованы)
        knowledge_result = {
            "answer": f"Ответ на вопрос категории: {intent_result['label']}"
        }
        
        action_result = {
            "suggestions": [
                f"Действие 1 для категории {intent_result['label']} и эмоции {emotion_result['label']}",
                f"Действие 2 для категории {intent_result['label']}"
            ]
        }
        
        summary_result = {
            "summary": f"Диалог категории {intent_result['label']} с эмоциональной окраской {emotion_result['label']}",
            "crm_record": {
                "category": intent_result['label'],
                "emotion": emotion_result['label'],
                "status": "processed"
            }
        }
        
        quality_result = {
            "quality_score": 8.5,
            "feedback": "Обработка диалога выполнена удовлетворительно"
        }
        
        # Формируем общий результат
        result = {
            "input_dialog": dialog,
            "intent_analysis": intent_result,
            "emotion_analysis": emotion_result,
            "knowledge_response": knowledge_result,
            "action_suggestions": action_result,
            "summary": summary_result,
            "quality_assessment": quality_result
        }
        
        return result
    
    def process_dialog(self, dialog: List[str]) -> Dict[str, Any]:
        """
        Обрабатывает диалог с использованием CrewAI
        
        Args:
            dialog: Список сообщений диалога
            
        Returns:
            Словарь с результатами обработки диалога всеми агентами
        """
        try:
            print("Запуск обработки диалога с использованием CrewAI...")
            
            # Подготавливаем входные данные для CrewAI
            inputs = {
                "dialog": dialog,
                "user_text": self._extract_user_messages(dialog)
            }
            
            # Запускаем обработку с использованием CrewAI
            crew_result = self.crew.kickoff(inputs=inputs)
            
            # Обрабатываем результат
            # Примечание: формат результата может зависеть от реализации CrewAI и может потребовать дополнительной обработки
            processed_result = self._process_crew_result(crew_result, dialog)
            
            return processed_result
            
        except Exception as e:
            print(f"Ошибка при обработке диалога с использованием CrewAI: {e}")
            print("Использование резервного метода обработки с фактическими моделями...")
            
            # Используем резервный метод обработки
            return self.process_dialog_with_actual_models(dialog)
    
    def _process_crew_result(self, crew_result, dialog: List[str]) -> Dict[str, Any]:
        """
        Обрабатывает результат CrewAI
        
        Args:
            crew_result: Результат от crew.kickoff()
            dialog: Оригинальный диалог
            
        Returns:
            Структурированный словарь с результатами
        """
        # Примечание: эта функция должна быть адаптирована под фактический формат результата CrewAI
        
        try:
            # Если результат - строка, пытаемся преобразовать его в JSON
            if isinstance(crew_result, str):
                try:
                    parsed_result = json.loads(crew_result)
                except json.JSONDecodeError:
                    parsed_result = {"raw_result": crew_result}
            else:
                parsed_result = crew_result
            
            # Создаем структурированный результат
            structured_result = {
                "input_dialog": dialog,
                "intent_analysis": parsed_result.get("intent_analysis", {}),
                "emotion_analysis": parsed_result.get("emotion_analysis", {}),
                "knowledge_response": parsed_result.get("knowledge_response", {}),
                "action_suggestions": parsed_result.get("action_suggestions", {}),
                "summary": parsed_result.get("summary", {}),
                "quality_assessment": parsed_result.get("quality_assessment", {})
            }
            
            return structured_result
            
        except Exception as e:
            print(f"Ошибка при обработке результата CrewAI: {e}")
            
            # В случае ошибки возвращаем результат с ошибкой
            return {
                "input_dialog": dialog,
                "error": f"Ошибка при обработке результата CrewAI: {str(e)}",
                "raw_result": str(crew_result)
            }


# Пример использования
if __name__ == "__main__":
    # Создаем экипаж
    crew_manager = DialogCrew()
    
    # Обрабатываем тестовый диалог
    test_dialog = [
        "Здравствуйте, я хочу узнать о ваших бизнес-решениях",
        "Добрый день! Конечно, я расскажу вам о наших бизнес-решениях.",
        "Меня интересуют решения для малого бизнеса",
        "У нас есть специальные предложения для малого бизнеса."
    ]
    
    # Обрабатываем диалог
    result = crew_manager.process_dialog(test_dialog)
    
    # Выводим результат
    print("\nРезультат обработки диалога:")
    print(json.dumps(result, ensure_ascii=False, indent=2))