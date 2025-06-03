from ui.core.base import BaseController
from ui.models.stats_model import StatsModel
from ui.views.stats_view import StatsView
from typing import Dict, Any

class StatsController(BaseController):
    """Controller for the stats page."""
    
    def __init__(self):
        self.model = StatsModel()
        self.view = StatsView()
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Initialize the stats page."""
        data = self.model.get_data()
        self.view.update(data)
        self.view.render()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle user input (not needed for stats page yet)."""
        pass 