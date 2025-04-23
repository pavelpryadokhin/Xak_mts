from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import sys

# Импортируем менеджер CrewAI
from crew_manager import DialogCrew

# Модель запроса
class DialogRequest(BaseModel):
    dialog: List[str]

# Создаем FastAPI приложение
app = FastAPI(
    title="Crew Dialog Processing API",
    description="API для обработки диалогов с использованием CrewAI",
    version="1.0.0"
)

# Глобальный экземпляр менеджера CrewAI
crew_manager = None

@app.on_event("startup")
async def startup_event():
    """Инициализация менеджера CrewAI при запуске сервера"""
    global crew_manager
    try:
        crew_manager = DialogCrew()
        print("Менеджер CrewAI успешно инициализирован")
    except Exception as e:
        print(f"Ошибка при инициализации менеджера CrewAI: {e}")
        print("API может работать некорректно!")

def get_crew_manager():
    """Получение экземпляра менеджера CrewAI"""
    if crew_manager is None:
        raise HTTPException(
            status_code=500, 
            detail="Менеджер CrewAI не инициализирован"
        )
    return crew_manager

@app.post("/process")
async def process_dialog(request: DialogRequest, manager: DialogCrew = Depends(get_crew_manager)):
    """
    Обрабатывает диалог с использованием CrewAI
    
    Принимает список сообщений диалога и возвращает результаты обработки всеми агентами
    """
    if not request.dialog:
        raise HTTPException(status_code=400, detail="Диалог не может быть пустым")
    
    try:
        # Обрабатываем диалог
        result = manager.process_dialog(request.dialog)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке диалога: {str(e)}"
        )

@app.post("/intent_agent/result")
async def predict_intent(request: DialogRequest, manager: DialogCrew = Depends(get_crew_manager)):
    """
    Предсказывает интент диалога
    
    Принимает список сообщений диалога и возвращает только результат Intent Agent
    """
    if not request.dialog:
        raise HTTPException(status_code=400, detail="Диалог не может быть пустым")
    
    try:
        # Извлекаем сообщения пользователя
        user_text = manager._extract_user_messages(request.dialog)
        
        # Получаем предсказание
        result = manager.intent_predictor.predict(user_text)
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании интента: {str(e)}"
        )

@app.post("/emotion_agent/result")
async def predict_emotion(request: DialogRequest, manager: DialogCrew = Depends(get_crew_manager)):
    """
    Определяет эмоциональный тон диалога
    
    Принимает список сообщений диалога и возвращает только результат Emotion Agent
    """
    if not request.dialog:
        raise HTTPException(status_code=400, detail="Диалог не может быть пустым")
    
    try:
        # Извлекаем сообщения пользователя
        user_text = manager._extract_user_messages(request.dialog)
        
        # Получаем предсказание
        raw_result = manager.emotion_predictor.predict(user_text)
        
        # Форматируем результат
        result = {
            'label': raw_result[0]['label'],
            'score': float(raw_result[0]['score'])
        }
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при определении эмоций: {str(e)}"
        )

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "message": "Crew Dialog Processing API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/process", "method": "POST", "description": "Полная обработка диалога всеми агентами с использованием CrewAI"},
            {"path": "/intent_agent/result", "method": "POST", "description": "Предсказание интента диалога"},
            {"path": "/emotion_agent/result", "method": "POST", "description": "Определение эмоционального тона диалога"}
        ]
    }

@app.get("/health")
async def health_check():
    """Проверка состояния API"""
    global crew_manager
    
    status = "healthy" if crew_manager is not None else "unhealthy"
    agents = []
    
    if crew_manager is not None:
        if hasattr(crew_manager, "intent_predictor"):
            agents.append("Intent Agent")
        if hasattr(crew_manager, "emotion_predictor"):
            agents.append("Emotion Agent")
    
    return {
        "status": status,
        "loaded_agents": agents,
        "crewai_available": crew_manager is not None
    }

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run("crew_api:app", host="0.0.0.0", port=8000, reload=True)