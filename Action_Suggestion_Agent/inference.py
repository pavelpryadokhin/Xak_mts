import requests
from typing import Dict
import json
import copy
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

PROMPT = """
[ROLE]
Ты — опытный специалист по поддержке клиентов колл-центра с более чем 15-летним стажем.
Твоя задача — создавать максимально персонализированные, практичные и эмпатичные рекомендации 
для оператора, основываясь на полном спектре предоставленных данных.

[КОНТЕКСТ]
1. Тема вопроса клиента: {intent} (достоверность: {confidence})
2. Эмоция: {emotion} (достоверность: {confidence_1})
3. Метаданные:
    - Тариф: {tariff}
    - Устройство: {device}
    - Активные сервисы: {active_services}
    - Финансовые продукты: {financial}
    - Подписки: {subscriptions}

[ИНСТРУКЦИЯ]
Создать рекомендации, которые должны включать:

**A. Персонализированное приветствие и установление контакта:**
- Учитывать тему обращения, эмоциональное состояние и историю взаимодействия
- Использовать имя клиента и упоминать его услуги или особенности (например, "Уважаемый Андрей, вы давно пользуетесь нашим тарифом STANDARD...")
- Приветствие должно отображать эмпатию и желание помочь

**B. Анализ ситуации и ключевые рекомендации:**
- Предложить конкретные решения или идеи, исходя из точных данных (например, для активных подписок — предложить апгрейд или новый продукт)
- Учитывать устройство: рекомендации, связанные с его особенностями или возможностями
- Учитывать эмоциональный настрой: использовать мягкий или мотивирующий тон

**C. Персональные советы и предложения:**
- Включать рекомендации по оптимизации тарифов, предложений по бонусам или новым услугам, релевантным именно этому клиенту
- Предлагать дополнительные услуги, которые соответствуют его текущим потребностям и поведению
- Предлагать техническую помощь, исходя из устройства и используемых сервисов

**D. Эмоциональное и психологическое взаимодействие:**
- Поддерживать позитив, снижать напряжение и создавать ощущение заботы
- Учитывать возможные боли или опасения клиента (например, стоимость, сложность настройки и т.п.)

**[Пример для клиента с активным тарифом и положительной историей]**
"Добрый день, Александр! Благодарим за долгую работу с нами. Рекомендуем ознакомиться с нашими новыми предложениями по улучшению вашего тарифа, а также советуем активировать дополнительные услуги для повышения удобства использования."

[Формат ответа]
Ответ оформляй в виде структурированного списка с разделами:
• Персонализированное приветствие
• Анализ ситуации и рекомендации
• Персонализированные предложения
• Эмоциональный и психологический совет
    """

DEFAULT_SCHEMA = {
    "subscriber_metadata": {
        "phone": "",
        "operator": {"is_mts": False},
        "tariff": {"name": "", "includes": []},
        "services": {
            "mobile": False,
            "home_internet": False,
            "home_tv": False,
            "home_phone": False
        },
        "device": {
            "model": "",
            "os": {"name": "", "version": ""}
        },
        "applications": {
            "my_mts": False,
            "lk": False,
            "mts_bank": False,
            "mts_money": False
        },
        "subscriptions": {
            "mts_premium": False,
            "mts_cashback": False,
            "zashchitnik_basic": False,
            "zashchitnik_plus": False,
            "kion": False,
            "music": False,
            "stroki": False
        },
        "financial_products": {
            "mts_bank": {"debit_card": False, "credit_card": False},
            "mts_money": {"debit_card": False, "credit_card": False, "virtual_card": False}
        }
    }
}




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
            "max_tokens": 400,
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
            print(error_msg)
            return 'Ошибка при генерации ответа'


class ActionSuggestionAgent:
    def __init__(self, intent_agent, emotion_agent, llm: CompletionAPI):
        """
        Инициализация агента рекомендаций
        
        :param intent_agent: Объект Intent Agent с атрибутом intent_data
        :param emotion_agent: Объект Emotion Agent с атрибутом emotion_data
        :param llm: Экземпляр CompletionAPI для генерации текста
        """
        self.intent_agent = intent_agent
        self.emotion_agent = emotion_agent
        self.llm = llm

    def load_and_fix_json(self,file_path: str) -> dict:
        """Загружает JSON файл и добавляет недостающие поля с пустыми значениями"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        default_schema = DEFAULT_SCHEMA

        def _merge_nested(current: dict, schema: dict) -> None:
            for key, default_val in schema.items():
                if key not in current:
                    current[key] = copy.deepcopy(default_val)
                elif isinstance(default_val, dict):
                    if not isinstance(current[key], dict):
                        current[key] = copy.deepcopy(default_val)
                    else:
                        _merge_nested(current[key], default_val)
                elif isinstance(default_val, list) and not current[key]:
                    current[key] = copy.deepcopy(default_val)

        if 'subscriber_metadata' not in data:
            data['subscriber_metadata'] = copy.deepcopy(default_schema['subscriber_metadata'])
        else:
            _merge_nested(data['subscriber_metadata'], default_schema['subscriber_metadata'])
        
        return data["subscriber_metadata"]


    def _prepare_prompt(self, file_path: str) -> str:
        metadata = self.load_and_fix_json(file_path)
        prompt_template = PROMPT

        # Парсинг метаданных
        parsed_meta = {
            "tariff": metadata['tariff']['name'],
            "device": f"{metadata['device']['model']} ({metadata['device']['os']['name']})",
            "active_services": ", ".join([k for k,v in metadata['services'].items() if v]),
            "financial": ", ".join([
                *[f"Дебетовая карта МТС Банк" if metadata['financial_products']['mts_bank']['debit_card'] else ""],
                *[f"Premium подписка" if metadata['subscriptions']['mts_premium'] else ""]
            ]),
            "subscriptions": ", ".join([k for k,v in metadata['subscriptions'].items() if v])
        }

        return prompt_template.format(
            intent=self.intent_agent['label'],
            confidence=self.intent_agent['confidence'],
            emotion=self.emotion_agent['label'],
            confidence_1 = self.emotion_agent['score'],
            **parsed_meta
        )


    def generate_recommendations(self,file_path):
        """Генерация финальных рекомендаций с помощью LLM"""
        prompt = self._prepare_prompt(file_path)
        return self.llm.generate(prompt)

# Пример использования
if __name__ == "__main__":
    # Инициализация компонентов
    llm = CompletionAPI(
        os.getenv('COMPLETION_API_URL'),
        os.getenv('COMPLETION_API_KEY'),
        os.getenv('COMPLETION_MODEL'),
    )
    intent_agent = {
            'label': 'Бизнес-решения и корпоративные услуги',
            'confidence': 0.3561,
            'db_categories':'id_category_2'
        }
    emotion_agent = {'label':'no_emotion','score':'0.6442'}
    action_agent = ActionSuggestionAgent(
        intent_agent,
        emotion_agent,
        llm
    )
    
    # Генерация рекомендаций
    recommendations = action_agent.generate_recommendations('Data/meta_inf_2.json')
    print("Рекомендации для оператора:")
    print(recommendations)
