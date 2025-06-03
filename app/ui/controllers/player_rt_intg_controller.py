from ui.core.base import BaseController
from ui.models.player_rt_intg_model import PlayerRtIntgModel
from ui.views.player_rt_intg_view import PlayerRtIntgView
from typing import Dict, Any

class PlayerRtIntgController(BaseController):
    """Controller for the player ratings (integral method) page."""
    
    def __init__(self):
        self.model = PlayerRtIntgModel()
        self.view = PlayerRtIntgView()
        super().__init__(self.model, self.view)

    def initialize(self) -> None:
        """Initialize the player ratings page."""
        data = self.model.get_data()
        self.view.update(data)
        self.view.render()

    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle user input (not needed for this page yet)."""
        pass 