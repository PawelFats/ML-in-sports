from app.ui.core.base import BaseController
from app.ui.models.player_rt_red_model import PlayerRtRedModel
from app.ui.views.player_rt_red_view import PlayerRtRedView
from typing import Dict, Any

class PlayerRtRedController(BaseController):
    """Контроллер для страницы рейтинга игроков (советский метод).
    Отвечает за взаимодействие модели и представления, связанных с советским способом оценки."""

    def __init__(self):
        # Создаем экземпляры модели и представления
        self.model = PlayerRtRedModel()       # Модель использует функцию player_rt_red() для генерации данных
        self.view = PlayerRtRedView()         # Представление отображает результат в интерфейсе
        super().__init__(self.model, self.view)  # Передаём их в базовый контроллер

    def initialize(self) -> None:
        """Инициализация страницы рейтингов игроков."""
        data = self.model.get_data()  # Получаем данные из модели
        self.view.update(data)        # Передаём их в представление
        self.view.render()            # Отрисовываем страницу

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка пользовательского ввода (на текущей странице не требуется)."""
        pass  # Ввод со стороны пользователя пока не используется
