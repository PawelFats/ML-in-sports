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
        st.sidebar.title("Меню")
        if st.session_state.get("lock_navigation", False):
            st.sidebar.warning("⏳ Дождитесь завершения обработки данных")

        # Добавляем disabled
        self.current_page = st.sidebar.radio(
            "Выберите раздел:",
            self.pages,
            index=self.pages.index(self.current_page),
            disabled=st.session_state.get("lock_navigation", False),
        )


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
        # 1. Настраиваем Streamlit
        self.setup()

        # 2. Инициализируем флаг блокировки (если ещё нет)
        if "lock_navigation" not in st.session_state:
            st.session_state.lock_navigation = False

        # 3. Сначала рисуем сайдбар — он сразу увидит lock_navigation и отобразит либо радио, либо предупреждение
        self.render_sidebar()

        # 4. Затем запускаем контроллер для выбранной страницы
        if self.current_page in self.controllers:
            controller = self.controllers[self.current_page]
            controller.initialize()
        else:
            st.error(f"Controller for page '{self.current_page}' not found!")

        # 5. И в самом конце — футер
        self.render_footer()

