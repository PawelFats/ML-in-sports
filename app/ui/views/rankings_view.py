import streamlit as st
from ui.core.base import BaseView
from typing import Any, Dict, Optional
import pandas as pd

class RankingsView(BaseView):
    """View for the team rankings page."""
    
    def __init__(self):
        self.title: str = ""
        self.description: str = ""
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """Update the view data."""
        self.title = data.get("title", "")
        self.description = data.get("description", "")
        self.data = data.get("data")
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """Render the rankings page."""
        st.title(self.title)
        st.write(self.description)
        
        if self.error_message:
            st.error(self.error_message)
        elif self.data is not None:
            st.dataframe(self.data) 