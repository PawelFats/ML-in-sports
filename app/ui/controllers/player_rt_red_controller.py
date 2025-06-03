from ui.core.base import BaseController
from ui.models.player_rt_red_model import PlayerRtRedModel
from ui.views.player_rt_red_view import PlayerRtRedView
from typing import Dict, Any

class PlayerRtRedController(BaseController):
    """Controller for the player ratings (soviet method) page."""
    
    def __init__(self):
        self.model = PlayerRtRedModel()
        self.view = PlayerRtRedView()
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Initialize the player ratings page."""
        data = self.model.get_data()
        self.view.update(data)
        self.view.render()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle user input (not needed for this page yet)."""
        pass 