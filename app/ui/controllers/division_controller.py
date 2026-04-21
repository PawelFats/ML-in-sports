import streamlit as st
from ui.core.base import BaseController
from ui.models.division_model import DivisionModel
from ui.views.division_view import DivisionView
from typing import Dict, Any

class DivisionController(BaseController):
    """Контроллер для страницы распределения команд по дивизионам."""
    
    def __init__(self):
        self.model = DivisionModel()
        self.view = DivisionView()
        super().__init__(self.model, self.view)
    
    def initialize(self) -> None:
        """Инициализация страницы распределения команд."""
        # Инициализируем данные по умолчанию
        data = {
            'evaluation_result': {},
            'ranking_result': {},
            'custom_evaluation_result': {}
        }
        
        # Отрисовываем view - он сам обработает нажатия кнопок через callback
        self.view.update(data)
        self.view.render()
        
        # Обрабатываем результаты после рендеринга
        if 'evaluation_result' in st.session_state:
            data['evaluation_result'] = st.session_state.evaluation_result
            self.view.update(data)
            self.view.render()
        
        if 'ranking_result' in st.session_state:
            data['ranking_result'] = st.session_state.ranking_result
            self.view.update(data)
            self.view.render()
        
        if 'custom_evaluation_result' in st.session_state:
            data['custom_evaluation_result'] = st.session_state.custom_evaluation_result
            self.view.update(data)
            self.view.render()
    
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка пользовательского ввода."""
        # Обработка ввода происходит через initialize() при нажатии кнопок
        pass

