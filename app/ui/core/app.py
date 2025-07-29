import streamlit as st  # библиотека для веб-интерфейсов
from typing import Dict  # аннотация типов для словаря
from ui.core.base import BaseController  # базовый класс контроллеров приложения

class HockeyAnalyticsApp:
    """Главный класс приложения, управляет состоянием и маршрутизацией страниц."""

    def __init__(self):
        # Словарь, где ключ — название страницы, значение — её контроллер
        self.controllers: Dict[str, BaseController] = {}
        # Текущая выбранная пользователем страница (по умолчанию "Главная")
        self.current_page: str = "Главная"
        # Список доступных страниц приложения
        self.pages = [
            "Главная",
            "Таблица с данными за каждую игру",
            "Актуальные рейтинги команд",
            "ELO рейтинг команды за все время",
            "Рейтинг игроков (интегральный метод)",
            "Рейтинг игроков (советский метод)",
            "Байесовский метод"
        ]

    def register_controller(self, page_name: str, controller: BaseController) -> None:
        """Регистрация контроллера для конкретной страницы по её имени."""
        self.controllers[page_name] = controller

    def setup(self) -> None:
        """Инициализация конфигурации Streamlit (заголовок, макет)."""
        st.set_page_config(
            page_title="Анализ хоккейной статистики",  # заголовок вкладки браузера
            layout="wide"  # широкая вёрстка для удобства отображения таблиц и графиков
        )

    def render_sidebar(self) -> None:
        """Отрисовка бокового меню с выбором страницы."""
        st.sidebar.title("Меню")  # заголовок боковой панели
        # Радио-кнопки для выбора текущей страницы из списка
        self.current_page = st.sidebar.radio("Выберите раздел:", self.pages)

    def render_footer(self) -> None:
        """Отрисовка нижнего колонтитула с информацией об авторах."""
        st.markdown("---")  # горизонтальная линия
        # Информация об авторе и помощь
        st.markdown(
            "<p style='text-align: center; color: grey;'>"
            "© 2025 Project Author: Pavel Fatyanov, Novosibirsk State Technical University"
            "</p>",
            unsafe_allow_html=True  # разрешаем HTML для стилизации
        )
        st.markdown(
            "<p style='text-align: center; color: grey;'>"
            "Assistance provided by Maxim Bakaev and Elizaveta Ulederkina."
            "</p>",
            unsafe_allow_html=True
        )

    def run(self) -> None:
        """Запуск приложения: настройка, меню, отрисовка выбранной страницы и футер."""
        # Настраиваем Streamlit
        self.setup()
        # Рисуем боковое меню и получаем выбранную страницу
        self.render_sidebar()
        
        # Если для выбранной страницы зарегистрирован контроллер, запускаем его
        if self.current_page in self.controllers:
            controller = self.controllers[self.current_page]
            controller.initialize()  # инициализация контроллера отображает View
        else:
            # В случае отсутствия контроллера — выводим ошибку
            st.error(f"Controller for page '{self.current_page}' not found!")
        
        # Рисуем нижний колонтитул
        self.render_footer()
