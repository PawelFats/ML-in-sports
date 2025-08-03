from app.ui.core.base import BaseController  # базовый класс контроллеров
from app.ui.models.bayesian_model import BayesianModel  # модель для байесовского анализа
from app.ui.views.bayesian_view import BayesianView  # представление для байесовского анализа
from typing import Dict, Any  # аннотации типов

class BayesianController(BaseController):
    """Контроллер для страницы байесовского анализа."""

    def __init__(self):
        # Инициализируем модель и представление
        self.model = BayesianModel()
        self.view = BayesianView()
        # Вызываем конструктор базового класса с моделью и view
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Начальная инициализация страницы байесовского анализа.
        Получаем данные из модели, обновляем представление и отрисовываем его."""
        # Запрашиваем данные у модели (включая расчёт, если нужно)
        data = self.model.get_data()
        # Передаём полученные данные во view
        self.view.update(data)
        # Отрисовываем интерфейс представления
        self.view.render()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработать ввод пользователя (пока не используется на этой странице)."""
        pass
