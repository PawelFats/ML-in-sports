import pandas as pd  # библиотека для работы с табличными данными
from ui.core.base import BaseModel  # базовый класс для моделей приложения
from typing import Any, Optional  # аннотации типов
from pathlib import Path  # для работы с файловыми путями


class RankingsModel(BaseModel):
    """Модель для страницы рейтинга команд."""

    def __init__(self):
        # Таблица с рейтингами команд, читаемая из CSV
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при загрузке данных
        self.error_message: Optional[str] = None
        # Путь к файлу с объединёнными рейтингами команд
        self.data_path = Path("data/targeted/team_ratings_merge.csv")

    def load_data(self) -> None:
        """Загрузка данных о рейтингах команд из CSV-файла."""
        try:
            # Считываем CSV в DataFrame
            self.data = pd.read_csv(self.data_path)
            # Очищаем сообщение об ошибке при успешной загрузке
            self.error_message = None
        except Exception as e:
            # В случае ошибки обнуляем данные и сохраняем текст ошибки
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_data(self) -> dict[str, Any]:
        """Возвращает текущее состояние модели для контроллера."""
        # Если данные ещё не загружены, выполняем загрузку
        if self.data is None:
            self.load_data()

        # Формируем словарь с данными для представления
        return {
            "title": "Рейтинги команд",  # заголовок страницы
            "description": "Здесь будет информация о рейтингах команд.",  # описание под заголовком
            "data": self.data,  # DataFrame с рейтингами
            "error_message": self.error_message  # возможное сообщение об ошибке
        }

    def update(self, data: Any) -> None:
        """Метод обновления модели (не используется для страницы рейтингов команд)."""
        pass