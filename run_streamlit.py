#!/usr/bin/env python3
"""
Скрипт для запуска Streamlit приложения с оптимизированными параметрами
для предотвращения WebSocket ошибок и улучшения производительности.
"""

import subprocess
import sys
import os

def main():
    # Параметры для запуска Streamlit
    streamlit_params = [
        sys.executable, "-m", "streamlit", "run",
        "app/ui/main.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.maxUploadSize", "200",
        "--server.enableXsrfProtection", "false",
        "--server.enableCORS", "false",
        "--browser.gatherUsageStats", "false",
        "--client.showErrorDetails", "true",
        "--client.toolbarMode", "minimal",
        "--runner.magicEnabled", "true",
        "--logger.level", "info"
    ]
    
    print("Запуск Streamlit приложения с оптимизированными параметрами...")
    print("Параметры:", " ".join(streamlit_params[4:]))  # Показываем только параметры
    
    try:
        # Запускаем Streamlit
        subprocess.run(streamlit_params, check=True)
    except KeyboardInterrupt:
        print("\nПриложение остановлено пользователем")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка запуска Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
