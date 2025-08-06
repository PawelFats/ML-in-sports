from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModel(ABC):
    """Базовый класс для всех моделей в приложении.
    Определяет интерфейс для получения и обновления данных модели."""

    @abstractmethod
    def get_data(self) -> Any:
        """Получить данные из модели.
        Должен возвращать результат работы модели (например, словарь, DataFrame или другую структуру)."""
        pass

    @abstractmethod
    def update(self, data: Any) -> None:
        """Обновить внутренние данные модели на основе входных данных.
        Например, изменить состояние модели в ответ на действия пользователя."""
        pass

class BaseView(ABC):
    """Базовый класс для всех представлений (View) в приложении.
    Задаёт интерфейс для обновления данных и отрисовки UI во фреймворке Streamlit."""

    @abstractmethod
    def update(self, data: Any) -> None:
        """Обновить внутреннее состояние представления перед отрисовкой.
        Принимает данные, подготовленные контроллером или моделью."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Отрендерить текущее состояние представления.
        Здесь вызываем Streamlit-функции для отображения заголовков, таблиц, графиков и т.д."""
        pass

class BaseController(ABC):
    """Базовый класс для всех контроллеров (Controller) в приложении.
    Отвечает за взаимодействие между моделью и представлением, обработку пользовательского ввода."""

    def __init__(self, model: BaseModel, view: BaseView):
        # Сохраняем ссылки на модель и представление
        self.model = model
        self.view = view

    @abstractmethod
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработать ввод от пользователя.
        Полученные данные передать в модель, затем обновить представление."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Начальная инициализация контроллера.
        Должна получить данные из модели, передать их во view и вызвать render()."""
        pass
