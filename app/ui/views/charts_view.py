import streamlit as st
import plotly.express as px
from ui.core.base import BaseView
from typing import Any, Dict, Optional, List
import pandas as pd

class ChartsView(BaseView):
    """View for the team ELO charts page."""
    
    def __init__(self):
        self.title: str = ""
        self.data: Optional[pd.DataFrame] = None
        self.teams: List[int] = []
        self.selected_team_data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """Update the view data."""
        self.title = data.get("title", "")
        self.data = data.get("data")
        self.teams = data.get("teams", [])
        self.selected_team_data = data.get("selected_team_data")
        self.error_message = data.get("error_message")

    def render(self) -> None:
        st.title(self.title)

        if self.error_message:
            st.error(self.error_message)
            return

        selected_team = st.selectbox(
            "Выберите команду:",
            self.teams,
            key="selected_team"
        )

        if self.selected_team_data is not None and not self.selected_team_data.empty:
            self.selected_team_data = self.selected_team_data.copy()
            self.selected_team_data["date"] = pd.to_datetime(self.selected_team_data["date"])
            fig = px.line(
                self.selected_team_data,
                x="date",
                y="ELO",
                title=f"Рейтинг ELO команды {selected_team} по времени"
            )
            st.plotly_chart(fig)
        elif self.selected_team_data is not None:
            st.warning("Нет данных для выбранной команды.")

