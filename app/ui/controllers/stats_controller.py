from ui.core.base import BaseController
from ui.models.stats_model import StatsModel
from ui.views.stats_view import StatsView
from typing import Dict, Any

class StatsController(BaseController):
    """Контроллер для страницы со статистикой команд.
    Отвечает за получение данных о матчах и их отображение."""

    def __init__(self):
        # Инициализируем модель и представление для страницы статистики
        self.model = StatsModel()   # Модель загружает данные из CSV с игровой статистикой
        self.view = StatsView()     # Представление отвечает за отображение статистики в UI
        super().__init__(self.model, self.view)  # Передаем их в базовый контроллер

    def initialize(self) -> None:
        """Инициализация страницы статистики.
        Получаем данные из модели, обновляем представление и отрисовываем страницу."""
        data = self.model.get_data()    # Получаем данные, при необходимости загружаем их из файла
        self.view.update(data)          # Обновляем данные в представлении
        self.view.render()              # Отрисовываем интерфейс

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка пользовательского ввода (пока не используется на странице статистики)."""
        pass
