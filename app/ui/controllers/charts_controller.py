from ui.core.base import BaseController
from ui.models.charts_model import ChartsModel
from ui.views.charts_view import ChartsView
from typing import Dict, Any
import streamlit as st

class ChartsController(BaseController):
    """Controller for the team ELO charts page."""
    
    def __init__(self):
        self.model = ChartsModel()
        self.view = ChartsView()
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Initialize the charts page."""
        self.model.load_data()

        # Проверим: выбран ли уже team в session_state
        selected_team = st.session_state.get("selected_team")

        if selected_team is not None:
            self.model.update({"selected_team": selected_team})

        data = self.model.get_data()
        self.view.update(data)
        self.view.render()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle team selection."""
        self.model.update(input_data)
        data = self.model.get_data()
        self.view.update(data)
        self.view.render() 