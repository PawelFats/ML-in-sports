from ui.core.base import BaseController
from ui.models.home_model import HomeModel
from ui.views.home_view import HomeView
from typing import Dict, Any

class HomeController(BaseController):
    """Контроллер для главной страницы (home page).
    Отвечает за получение данных из модели и передачу их во view."""

    def __init__(self):
        # Создаем экземпляры модели и представления (view)
        self.model = HomeModel()     # Модель хранит приветственные данные
        self.view = HomeView()       # Представление отрисовывает приветственный текст
        super().__init__(self.model, self.view)  # Инициализируем базовый контроллер с моделью и вью

    def initialize(self) -> None:
        """Инициализация главной страницы."""
        data = self.model.get_data()  # Получаем данные из модели
        self.view.update(data)        # Передаем их во view
        self.view.render()            # Отрисовываем интерфейс

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка пользовательского ввода (на главной странице не требуется)."""
        pass  # Ввод от пользователя на этой странице не обрабатывается
