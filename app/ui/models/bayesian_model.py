from ui.core.base import BaseModel  # базовый класс для моделей приложения
from typing import Any, Optional  # аннотации типов
from src.method_bayesian import bayesian_analysis  # функция для расчёта байесовского анализа
import pandas as pd  # библиотека для работы с табличными данными


class BayesianModel(BaseModel):
    """Модель для страницы байесовского анализа."""

    def __init__(self):
        # Хранилище результатов анализа (DataFrame) или None, если не загружено
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при попытке загрузки данных
        self.error_message: Optional[str] = None

    def load_data(self) -> None:
        """Выполнить расчёт байесовского анализа и сохранить результат."""
        try:
            # Вызов внешней функции, которая возвращает DataFrame с результатами
            self.data = bayesian_analysis()
            # При успешном выполнении очищаем сообщение об ошибке
            self.error_message = None
        except Exception as e:
            # Если произошла ошибка, сбрасываем данные и сохраняем текст ошибки
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_data(self) -> dict[str, Any]:
        """
        Вернуть текущее состояние модели в виде словаря для представления.
        Загружает данные, если они ещё не загружены.
        """
        # Если данные ещё не рассчитаны, выполняем загрузку
        if self.data is None:
            self.load_data()
        # Формируем словарь, который отдаст контроллер во View
        return {
            "title": "Байесовский метод",
            "data": self.data,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """
        Метод обновления модели (не используется для этой страницы).
        Реализован пустым, так как входных данных от пользователя нет.
        """
        pass

