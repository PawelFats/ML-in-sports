from ui.core.base import BaseModel  # базовый класс для моделей приложения
from typing import Any, Optional  # аннотации типов
from src.generate_ratings_intg import *  # функция расчёта рейтинга игроков интегральным методом
import pandas as pd  # библиотека для работы с DataFrame


class PlayerRtIntgModel(BaseModel):
    """Модель для страницы рейтинга игроков (интегральный метод)."""

    def __init__(self):
        # Таблица с результатами расчётов рейтинга
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при загрузке или расчёте
        self.error_message: Optional[str] = None

    def load_data(self) -> None:
        """Выполнить расчёт рейтингов игроков интегральным методом."""
        try:
            # Вызов внешней функции, возвращающей DataFrame с рейтингами
            self.data = player_rt_intg()

            # При успешном выполнении очищаем сообщение об ошибке
            self.error_message = None
        except Exception as e:
            # В случае ошибки обнуляем данные и сохраняем текст ошибки
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_data(self) -> dict[str, Any]:
        """Возвращает текущие данные модели для передачи в представление."""
        # Если данные ещё не загружены, загружаем их
        if self.data is None:
            self.load_data()
        # Формируем словарь с заголовком, данными и возможным сообщением об ошибке
        return {
            "title": "Рейтинг игроков (интегральный метод)",
            "data": self.data,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """Метод обновления модели (не используется для этой страницы)."""
        pass

