import requests
from typing import List
import re
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

PROMPT_1 = """Проанализируйте диалог оператора с клиентом на соответствие стандартам общения. 
Критерии оценки:
1. Вежливость и использование этикетных формул
2. Четкое следование скрипту компании
3. Профессиональная лексика
4. Эмпатия и понимание проблемы
5. Корректное использование терминов

Диалог:
{dialog}

Проанализируйте каждую реплику оператора. Если найдены нарушения:
1. Укажите конкретную реплику с нарушением
2. Опишите тип нарушения
3. Предложите альтернативный вариант фразы
4. Дайте общие рекомендации по улучшению

Формат вывода:
Анализ нарушений:
- [Номер нарушения] Реплика: "..."
  Тип нарушения: ...
  Рекомендация: ...

Общие рекомендации:...
"""

PROMPT_2="""
На основе анализа диалога технической поддержки создай развернутый план улучшения навыков оператора. 
{analysis}

**Требования к ответу:**
1. Язык: Русский (для англоязычных терминов приводить перевод в скобках)
"""

class CompletionAPI:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str,):
        """Генерация текста через Completion API"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.5,
            "max_tokens": 300,
            "top_p": 0.9,
            "frequency_penalty": 0.8,
            "presence_penalty": 0.4,
            "stop": []
        }
        try:
            response = requests.post(
                self.api_url,
                json=data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()['choices'][0]['text'].strip()
        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {str(e)}"
            return ''


class QualityAssuranceAgent:
    def __init__(self, llm: CompletionAPI):
        self.llm = llm

    def get_improvement_plan(self, analysis: str):
        """Генерация плана улучшения на основе анализа"""
        improvement_prompt = PROMPT_2.format(analysis = analysis)
        return self.llm.generate(improvement_prompt)

    def analyze_dialog(self, dialog: List[str]):
        """Анализ диалога и генерация рекомендаций"""
        formatted_dialog ="\n".join([f"{'Клиент:' if i % 2 == 0 else 'Оператор:'} {msg}" for i, msg in enumerate(test_dialog)])
        prompt = PROMPT_1.format(dialog=formatted_dialog)
        
        analysis = self.llm.generate(prompt)
        #improvement_plan = self.get_improvement_plan(analysis)
        return {
            "analysis": analysis,
            #"improvement_plan":improvement_plan,
        }




# Пример использования
if __name__ == "__main__":
    # Инициализация компонентов
    llm = CompletionAPI(
        os.getenv('COMPLETION_API_URL'),
        os.getenv('COMPLETION_API_KEY'),
        os.getenv('COMPLETION_MODEL'),
    )
    qa_agent = QualityAssuranceAgent(llm)
    
    # Пример диалога
    test_dialog = [
        "У меня пропало соединение, и я не могу завершить платеж за услугу.",
        "Ну проверьте интернет и попробуйте ещё раз, это же очевидно!",
        "Время идет, а проблема остается, очень разочарована этим сервисом.",
        "Это ваши локальные проблемы, у нас всё работает."
    ]
    
    # Анализ диалога
    analysis_result = qa_agent.analyze_dialog(test_dialog)
    print("Результат анализа:")
    print(analysis_result)

    

