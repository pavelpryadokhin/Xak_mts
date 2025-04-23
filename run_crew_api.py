#!/usr/bin/env python3
import os
import sys
import uvicorn

# Добавляем корневую директорию проекта в путь импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Запуск Crew API сервера...")
    
    # Устанавливаем текущую директорию на директорию этого скрипта
    # Это необходимо для корректного импорта модулей
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run("crew_api:app", host="0.0.0.0", port=8000, reload=True)
