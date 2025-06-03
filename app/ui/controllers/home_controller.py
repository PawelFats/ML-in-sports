from ui.core.base import BaseController
from ui.models.home_model import HomeModel
from ui.views.home_view import HomeView
from typing import Dict, Any

class HomeController(BaseController):
    """Controller for the home page."""
    
    def __init__(self):
        self.model = HomeModel()
        self.view = HomeView()
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Initialize the home page."""
        data = self.model.get_data()
        self.view.update(data)
        self.view.render()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle user input (not needed for home page)."""
        pass 