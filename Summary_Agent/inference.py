import requests
from typing import Dict
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

PROMPT = """
Ты — эксперт по классификации клиентских диалогов.
Твоя задача — определить тип диалога, исходя из его содержания.
В конце напиши только одно слово, без добавлений, строго из списка:

- **compensation** — если диалог связан с извинениями, предложениями компенсаций, бонусов или возвратами,
- **escalation** — если диалог предполагает перевод вопроса ниже или выше, передачу специалисту более высокого уровня,
- **info_provided** — если диалог сводится к предоставлению клиенту информации, разъяснений или данных.

Перед тем, как дать ответ, я дважды проверь свои выводы.
Диалог:
"{dialog}"

Проверка:  
1. Анализируешь содержимое и развитие диалога.  
2. Делаешь вывод о типе.  
3. Выводишь только ОДНО из трех слов: **"compensation"**, **"escalation"**, **"info_provided"**.  

Ответь только одним словом.
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
            "temperature": 0.0,         
            "max_tokens": 3,             
            "top_p": 1.0,               
            "frequency_penalty": 0.0,    
            "presence_penalty": 0.0,
            "stop": ['\n']
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
            print(error_msg)
            return 'Ошибка при генерации ответа'

class SummaryAgent:
    def __init__(self, intent_agent, emotion_agent,dialog, llm: CompletionAPI):
        self.intent_agent = intent_agent
        self.emotion_agent = emotion_agent
        self.llm = llm
        self.dialog = dialog


    def generate_resolution(self,dialog):
        formatted_dialog ="\n".join([f"{'Клиент:' if i % 2 == 0 else 'Оператор:'} {msg}" for i, msg in enumerate(dialog)])
        prompt = PROMPT.format(dialog=formatted_dialog)
        answer = self.llm.generate(prompt)
        answer = re.sub(r'[^a-z_]+', '', answer.lower())
        valid_responses = ['compensation', 'escalation', 'info_provided']
        # Находим наиболее похожий ответ из допустимых вариантов
        if answer in valid_responses:
            return answer
        else:
            # Если точного совпадения нет, ищем наиболее похожий ответ
            max_similarity = 0
            best_match = None
            for response in valid_responses:
                # Простое сравнение по количеству совпадающих символов
                similarity = sum(c1 == c2 for c1, c2 in zip(answer, response)) / max(len(answer), len(response))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = response
            
            # Возвращаем наиболее похожий ответ, если он достаточно похож
            if max_similarity > 0.5 and best_match:
                return best_match

    def generate_crm_data(self):
        """
        Генерирует данные для CRM на основе диалога, используя LLM.      
        """
        resolution = self.generate_resolution(', '.join(self.dialog))
        crm_data = {
            "issue_type": self.intent_agent.get("label", None),
            "client_sentiment": self.emotion_agent.get("label", None),
            "resolution": resolution
        }
        return crm_data


# Пример использования
if __name__ == "__main__":
    # Инициализация компонентов
    llm = CompletionAPI(
        os.getenv('COMPLETION_API_URL'),
        os.getenv('COMPLETION_API_KEY'),
        os.getenv('COMPLETION_MODEL'),
    )
    intent_agent = {
            'label': 'Мобильная связь, интернет и ТВ для дома и бизнеса',
            'confidence': 0.3561,
            'db_categories':'id_category_2'
        }
    emotion_agent = {'label':'anger','score':'0.6442'}
    dialog =  [
      "У меня пропало соединение, и я не могу завершить платеж за услугу.",
      "Извините за неудобство. Я сейчас проверю ваш статус соединения и помогу с оплатой.",
      "Время идет, а проблема остается, очень разочарована этим сервисом."
    ]
    summary_agent = SummaryAgent(
        intent_agent,
        emotion_agent,
        dialog,
        llm
    )
    print(summary_agent.generate_crm_data())
    

