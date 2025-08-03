from app.ui.core.base import BaseController
from app.ui.models.charts_model import ChartsModel
from app.ui.views.charts_view import ChartsView
from typing import Dict, Any
import streamlit as st

class ChartsController(BaseController):
    """Контроллер для страницы с графиками ELO рейтингов команд."""

    def __init__(self):
        # Создаём модель и представление, затем передаём их базовому контроллеру
        self.model = ChartsModel()
        self.view = ChartsView()
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Инициализация страницы: загружаем данные, обрабатываем выбранную команду, передаём всё в представление."""
        self.model.load_data()  # Загружаем данные из CSV
        
        data = self.model.get_data()
        self.view.update(data)

        if st.button("🔁 Перерасчитать рейтинги"):
            with st.spinner('Пересчёт рейтингов... Подождите, пожалуйста'):
                self.model.recalculate_elo_data()
            if self.model.error_message:
                st.error(self.model.error_message)
            else:
                st.success("Рейтинги пересчитаны!")

        # Проверим: выбрана ли команда ранее (сохранилась в session_state)
        selected_team = st.session_state.get("selected_team")

        if selected_team is not None:
            # Если да — обновим модель выбранной командой
            self.model.update({"selected_team": selected_team})

        # Получаем текущие данные модели и передаём их в представление
        data = self.model.get_data()
        self.view.update(data)
        self.view.render()  # Отрисовка представления

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка пользовательского ввода — выбор команды."""
        self.model.update(input_data)  # Обновляем модель новыми данными
        data = self.model.get_data()   # Получаем обновлённое состояние
        self.view.update(data)        # Передаём обновлённые данные во View
        self.view.render()            # Отрисовываем View
