#!/usr/bin/env python3
"""
Скрипт для обучения моделей машинного обучения для прогнозирования результатов матчей.
Сохраняет лучшую модель в файл MODEL.pkl для использования в алгоритме распределения команд.
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Импорт функции обучения моделей
try:
    from app.src.models import train_models
except ImportError:
    from src.models import train_models

def main():
    """Основная функция для обучения моделей."""
    print("=" * 60)
    print("Обучение моделей машинного обучения")
    print("=" * 60)
    
    # Параметры обучения
    cutoff_date = '2024-10-28'  # Дата разделения на обучающую и тестовую выборки
    game_stats_path = Path("data/targeted/game_stats_one_r.csv")
    model_save_path = Path("MODEL.pkl")
    
    # Проверка существования файла с данными
    if not game_stats_path.exists():
        print(f"Ошибка: Файл {game_stats_path} не найден!")
        print(f"Убедитесь, что файл существует по указанному пути.")
        sys.exit(1)
    
    print(f"\nПараметры обучения:")
    print(f"  - Дата разделения: {cutoff_date}")
    print(f"  - Файл данных: {game_stats_path}")
    print(f"  - Путь сохранения модели: {model_save_path}")
    print("\nНачинаем обучение моделей...")
    print("-" * 60)
    
    try:
        # Обучение моделей
        best_model = train_models(
            cutoff_date=cutoff_date,
            game_stats_path=str(game_stats_path),
            model_save_path=str(model_save_path)
        )
        
        print("-" * 60)
        print(f"\n✓ Обучение завершено успешно!")
        print(f"✓ Модель сохранена в: {model_save_path.absolute()}")
        print("\nМодель готова к использованию в алгоритме распределения команд.")
        
    except Exception as e:
        print(f"\n✗ Ошибка при обучении моделей: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

