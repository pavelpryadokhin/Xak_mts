from typing import List, Dict, Any, Union
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_qdrant import Qdrant
from langchain.schema.embeddings import Embeddings
from qdrant_client import QdrantClient
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

class CustomAPIEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для документов через API"""
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "input": texts
            },
            headers=self.headers
        )
        try:
            response.raise_for_status()
            data = response.json()
            return [item['embedding'] for item in data['data']]
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            return [[0.0] * 1024 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Генерация эмбеддинга для запроса через API"""
        return self.embed_documents([text])[0]


class CompletionAPI:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Генерация текста через Completion API"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.4,
            "max_tokens": 200,
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "stop": []
        }
        data.update({k: v for k, v in kwargs.items() if k != "stop_sequences"})
        
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
            return error_msg

class KnowledgeAgent:
    def __init__(self,
                 embedding_api_url: str,
                 embedding_api_key: str,
                 completion_api_url: str,
                 completion_api_key: str,
                 embedding_model: str = "bge-m3",
                 completion_model: str = "mws-gpt-alpha",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):

        # Инициализация компонентов
        self.embeddings = CustomAPIEmbeddings(embedding_api_url, embedding_api_key, embedding_model)
        self.llm = CompletionAPI(completion_api_url, completion_api_key, completion_model)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        self.prompt_template = self._create_prompt()

    def _get_retriever(self, db_key: str):
        return Qdrant(
            client=self.qdrant_client,
            collection_name=db_key,
            embeddings=self.embeddings
        ).as_retriever(search_kwargs={"k": 3})

    def _retrieve_documents(self, question: str, db_key: str) -> str:
        retriever = self._get_retriever(db_key)
        try:
            docs = retriever.get_relevant_documents(question)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Ошибка при получении документов: {str(e)}")
            return "Не удалось найти релевантные документы"

    def _create_prompt(self):
        template = """Вы являетесь помощником в выполнении заданий с ответами на вопросы.
Используй только приведенный контекст для ответа на вопрос.
Если вы не знаете ответа, просто скажите, что вы не знаете.
Если информации в контексте недостаточно, ответь "Не могу дать точный ответ".

Контекст: {context}

Вопрос: {question}

Ответ должен быть структурированным и содержательным. Используй маркированные списки где это уместно.
Ответ:"""
        return template

    def invoke(self, question: str, db_key: str) -> str:
        try:
            context = self._retrieve_documents(question, db_key)
            prompt = self.prompt_template.format(context=context, question=question)
            answer = self.llm.generate(prompt)
            
            return answer
        except Exception as e:
            print(f"Ошибка при вызове агента: {str(e)}")
            return f"Ошибка при вызове агента: {str(e)}"


# Инициализация агента из переменных окружения


# Тестовый запрос
if __name__ == "__main__":
    agent = KnowledgeAgent(
        embedding_api_url=os.getenv("EMBEDDING_API_URL"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
        completion_api_url=os.getenv("COMPLETION_API_URL"),
        completion_api_key=os.getenv("COMPLETION_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        completion_model=os.getenv("COMPLETION_MODEL"),
        qdrant_host=os.getenv("QDRANT_HOST"),
        qdrant_port=int(os.getenv("QDRANT_PORT"))
    )
    try:
        response = agent.invoke(
            question="как сделать перевод с карты на карту",
            db_key="id_category_0"
        )
        print("Ответ:")
        print(response)
    except Exception as e:
        print(f"Ошибка при выполнении тестового запроса: {str(e)}")