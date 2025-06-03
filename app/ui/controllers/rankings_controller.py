from ui.core.base import BaseController
from ui.models.rankings_model import RankingsModel
from ui.views.rankings_view import RankingsView
from typing import Dict, Any

class RankingsController(BaseController):
    """Controller for the team rankings page."""
    
    def __init__(self):
        self.model = RankingsModel()
        self.view = RankingsView()
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Initialize the rankings page."""
        data = self.model.get_data()
        self.view.update(data)
        self.view.render()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle user input (not needed for rankings page yet)."""
        pass 