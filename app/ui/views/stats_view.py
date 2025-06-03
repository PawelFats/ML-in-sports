import streamlit as st
from ui.core.base import BaseView
from typing import Any, Dict, Optional
import pandas as pd

class StatsView(BaseView):
    """View for the stats page."""
    
    def __init__(self):
        self.title: str = ""
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """Update the view data."""
        self.title = data.get("title", "")
        self.data = data.get("data")
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """Render the stats page."""
        st.title(self.title)
        
        if self.error_message:
            st.error(self.error_message)
        elif self.data is not None:
            st.write("Таблица статистики")
            st.dataframe(self.data) 