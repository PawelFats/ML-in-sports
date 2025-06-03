import streamlit as st
from typing import Dict, Type
from ui.core.base import BaseController

class HockeyAnalyticsApp:
    """Main application class that manages the application state and routing."""
    
    def __init__(self):
        self.controllers: Dict[str, BaseController] = {}
        self.current_page: str = "Главная"
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
        """Register a controller for a specific page."""
        self.controllers[page_name] = controller

    def setup(self) -> None:
        """Initialize the application configuration."""
        st.set_page_config(page_title="Анализ хоккейной статистики", layout="wide")

    def render_sidebar(self) -> None:
        """Render the application sidebar."""
        st.sidebar.title("Меню")
        self.current_page = st.sidebar.radio("Выберите раздел:", self.pages)

    def render_footer(self) -> None:
        """Render the application footer."""
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: grey;'>"
            "© 2025 Project Author: Pavel Fatyanov, Novosibirsk State Technical University"
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: grey;'>"
            "Assistance provided by Maxim Bakaev and Elizaveta Ulederkina."
            "</p>",
            unsafe_allow_html=True
        )

    def run(self) -> None:
        """Run the application."""
        self.setup()
        self.render_sidebar()
        
        # Render the current page
        if self.current_page in self.controllers:
            controller = self.controllers[self.current_page]
            controller.initialize()
        else:
            st.error(f"Controller for page '{self.current_page}' not found!")
        
        self.render_footer() 