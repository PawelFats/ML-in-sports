from ui.core.base import BaseController
from ui.models.rankings_model import RankingsModel
from ui.views.rankings_view import RankingsView
from typing import Dict, Any

class RankingsController(BaseController):
    """Контроллер для страницы с рейтингами команд.
    Отвечает за взаимодействие модели с рейтингами команд и их отображением."""

    def __init__(self):
        # Создаем экземпляры модели и представления для рейтингов команд
        self.model = RankingsModel()    # Модель загружает данные из CSV файла с рейтингами
        self.view = RankingsView()      # Представление отвечает за отрисовку данных в UI
        super().__init__(self.model, self.view)  # Передаем их в базовый контроллер

    def initialize(self) -> None:
        """Инициализация страницы рейтингов команд.
        Получаем данные из модели, обновляем представление и отрисовываем страницу."""
        data = self.model.get_data()    # Получаем данные из модели (загружаем из файла при необходимости)
        self.view.update(data)          # Передаем данные в представление
        self.view.render()              # Отрисовываем интерфейс

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка пользовательского ввода (пока не используется на этой странице)."""
        pass
