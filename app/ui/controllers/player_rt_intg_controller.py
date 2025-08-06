from ui.core.base import BaseController
from ui.models.player_rt_intg_model import PlayerRtIntgModel
from ui.views.player_rt_intg_view import PlayerRtIntgView
from typing import Dict, Any

class PlayerRtIntgController(BaseController):
    """Контроллер для страницы рейтинга игроков (интегральный метод).
    Связывает модель, содержащую данные, и представление, отображающее таблицу с рейтингами."""

    def __init__(self):
        # Инициализируем модель и представление для страницы
        self.model = PlayerRtIntgModel()     # Модель считает рейтинг игроков интегральным методом
        self.view = PlayerRtIntgView()       # Представление отображает рейтинг
        super().__init__(self.model, self.view)  # Вызываем конструктор базового контроллера

    def initialize(self) -> None:
        """Инициализация страницы рейтингов игроков."""
        data = self.model.get_data()  # Загружаем данные из модели
        self.view.update(data)        # Обновляем представление
        self.view.render()            # Отрисовываем интерфейс

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка пользовательского ввода (не требуется для этой страницы)."""
        pass  # Пользовательский ввод на этой странице пока не предусмотрен
