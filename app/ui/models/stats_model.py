import pandas as pd  # Импорт библиотеки pandas для работы с таблицами
from ui.core.base import BaseModel  # Импорт базовой модели из проекта
from typing import Any, Optional  # Типы для аннотаций
from pathlib import Path  # Для работы с файловыми путями

class StatsModel(BaseModel):
    """Модель для страницы статистики, отвечает за загрузку и предоставление данных о матчах."""

    def __init__(self):
        # Переменная для хранения загруженных данных (таблица pandas)
        self.data: Optional[pd.DataFrame] = None
        # Переменная для хранения текста ошибки, если что-то пойдёт не так
        self.error_message: Optional[str] = None
        # Путь к файлу с данными по статистике матчей
        self.data_path = Path("data/targeted/game_stats_one_r.csv")

    def load_data(self) -> None:
        """Загружает данные по матчам из CSV-файла."""
        try:
            # Пытаемся загрузить таблицу из CSV
            self.data = pd.read_csv(self.data_path)
            # Если успешно — очищаем сообщение об ошибке
            self.error_message = None
        except Exception as e:
            # В случае ошибки — очищаем данные
            self.data = None
            # Сохраняем текст ошибки для отображения во View
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_data(self) -> dict[str, Any]:
        """Возвращает словарь с текущими данными модели для использования в View."""
        # Если данные ещё не загружены — загружаем
        if self.data is None:
            self.load_data()
        
        # Возвращаем словарь с заголовком, данными и ошибкой (если есть)
        return {
            "title": "Статистика команд",  # Заголовок страницы
            "data": self.data,             # Загруженные данные
            "error_message": self.error_message  # Возможное сообщение об ошибке
        }

    def update(self, data: Any) -> None:
        """Метод обновления модели (для этой страницы не требуется, поэтому пустой)."""
        pass
