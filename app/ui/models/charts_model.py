import pandas as pd  # библиотека для работы с табличными данными
from ui.core.base import BaseModel  # базовый класс для моделей приложения
from typing import Any, Optional, List  # аннотации типов
from pathlib import Path  # для работы с путями файловой системы


class ChartsModel(BaseModel):
    """Модель для страницы графиков ELO рейтинга команды."""

    def __init__(self):
        # Полный DataFrame со статистикой игр (загружается из CSV)
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при загрузке данных
        self.error_message: Optional[str] = None
        # ID команды, выбранной пользователем
        self.selected_team: Optional[int] = None
        # Путь к файлу с данными о матчах
        self.data_path = Path("data/targeted/game_stats_one_r.csv")

    def load_data(self) -> None:
        """
        Загрузка данных из CSV-файла и первичная обработка.
        Считываем файл, конвертируем столбец даты в datetime и сортируем.
        """
        try:
            # Считываем CSV в DataFrame
            self.data = pd.read_csv(self.data_path)
            # Преобразуем столбец 'date' в datetime для последующей работы
            self.data["date"] = pd.to_datetime(self.data["date"])
            # Сортируем данные по дате по возрастанию
            self.data = self.data.sort_values(by="date", ascending=True)
            # Очищаем сообщение об ошибке при успешной загрузке
            self.error_message = None
        except Exception as e:
            # При ошибке обнуляем данные и сохраняем сообщение
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_team_list(self) -> List[int]:
        """
        Получить отсортированный список уникальных ID команд.
        Если данные ещё не загружены, сначала вызываем load_data().
        """
        if self.data is None:
            self.load_data()
        # Если данные есть, возвращаем уникальные ID команд, иначе пустой список
        return [] if self.data is None else sorted(self.data["ID team"].unique())

    def get_team_data(self, team_id: int) -> Optional[pd.DataFrame]:
        """
        Получить подтаблицу с данными по конкретной команде.
        Если данные не загружены, сначала загружаем их.
        """
        if self.data is None:
            self.load_data()
        # Фильтруем DataFrame по выбранному team_id
        return None if self.data is None else self.data[self.data["ID team"] == team_id]

    def get_data(self) -> dict[str, Any]:
        """
        Возвращает текущие данные модели в виде словаря для контроллера.
        Содержит:
        - title: заголовок страницы
        - data: полный DataFrame
        - teams: список доступных команд
        - selected_team_data: данные по выбранной команде или None
        - error_message: сообщение об ошибке
        """
        if self.data is None:
            self.load_data()
        return {
            "title": "Графики статистики",
            "data": self.data,
            # получаем список команд для виджета выбора
            "teams": self.get_team_list(),
            # данные для выбранной команды, если выбранная команда задана
            "selected_team_data": self.get_team_data(self.selected_team) if self.selected_team else None,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """
        Обновление состояния модели: здесь обновляется выбранная команда.
        Ожидается словарь с ключом 'selected_team'.
        """
        if isinstance(data, dict) and "selected_team" in data:
            self.selected_team = data["selected_team"]