from app.ui.core.base import BaseModel  # базовый класс для моделей приложения
from typing import Any, Optional  # аннотации типов
from app.src.generate_rating_red import player_rt_red  # функция расчёта рейтинга игроков советским методом
import pandas as pd  # библиотека для работы с табличными данными


class PlayerRtRedModel(BaseModel):
    """Модель для страницы рейтинга игроков ("советский" метод)."""

    def __init__(self):
        # Таблица с результатами расчётов рейтингов
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при загрузке или расчёте
        self.error_message: Optional[str] = None

    def load_data(self) -> None:
        """Выполнить расчёт рейтингов игроков советским методом."""
        try:
            # Вызов внешней функции, возвращающей DataFrame с рейтингами
            self.data = player_rt_red()
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
            "title": "Рейтинг игроков (советский метод)",
            "data": self.data,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """Метод обновления модели (не используется для этой страницы)."""
        pass
