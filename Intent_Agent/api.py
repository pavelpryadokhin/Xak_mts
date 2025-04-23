from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import sys

# Добавляем текущую директорию в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import IntentPredictor, db_categories

# Модель запроса
class DialogRequest(BaseModel):
    dialog: List[str]

# Создаем FastAPI приложение
app = FastAPI(title="Intent Prediction API")

# Инициализируем предиктор
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bert_model'))
predictor = IntentPredictor(model_path=model_path, labels_dict=db_categories)

@app.post("/intent_agent/result")
async def predict_intent(request: DialogRequest):
    """
    Предсказывает интент диалога
    
    Принимает список сообщений диалога и возвращает предсказанный интент
    """
    if not request.dialog:
        raise HTTPException(status_code=400, detail="Диалог не может быть пустым")
    
    text = " ".join([msg for i, msg in enumerate(request.dialog) if i % 2 == 0])
    
    # Получаем предсказание
    result = predictor.predict(text)
    
    return result

@app.get("/intent_agent")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "message": "Intent Prediction API",
        "endpoints": [
            {"path": "/intent_agent/result", "method": "POST", "description": "Предсказание интента диалога"}
        ]
    }

# Запуск сервера при непосредственном вызове файла
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 



