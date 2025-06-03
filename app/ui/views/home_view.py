import streamlit as st
from ui.core.base import BaseView
from typing import Any, Dict

class HomeView(BaseView):
    """View for the home page."""
    
    def __init__(self):
        self.data: Dict[str, str] = {}

    def update(self, data: Dict[str, str]) -> None:
        """Update the view data."""
        self.data = data

    def render(self) -> None:
        """Render the home page."""
        st.title(self.data.get("title", ""))
        st.write(self.data.get("description", ""))
        st.write(self.data.get("instruction", "")) 