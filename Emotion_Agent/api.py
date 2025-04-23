from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import sys

# Добавляем текущую директорию в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import EmotionPredictor

# Модель запроса
class DialogRequest(BaseModel):
    dialog: List[str]

# Создаем FastAPI приложение
app = FastAPI(title="Emotion Detection API")

# Инициализируем предиктор
predictor = EmotionPredictor()

@app.post("/emotion_agent/result")
async def predict_emotion(request: DialogRequest):
    """
    Определяет эмоциональный тон диалога
    
    Принимает список сообщений диалога и возвращает эмоциональную оценку
    """
    if not request.dialog:
        raise HTTPException(status_code=400, detail="Диалог не может быть пустым")
    
    # Объединяем сообщения диалога в один текст
    text = " ".join([msg for i, msg in enumerate(request.dialog) if i % 2 == 0])
    
    # Получаем предсказание
    raw_result = predictor.predict(text)
    
    # Форматируем результат
    result = {
        'label': raw_result[0]['label'],
        'score': float(raw_result[0]['score'])
    }
    
    return result

@app.get("/emotion_agent")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "message": "Emotion Detection API",
        "endpoints": [
            {"path": "/emotion_agent/result", "method": "POST", "description": "Определение эмоционального тона диалога"}
        ]
    }

# Запуск сервера при непосредственном вызове файла
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True) 