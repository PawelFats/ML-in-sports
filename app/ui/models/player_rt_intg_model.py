from ui.core.base import BaseModel
from typing import Any, Optional
from app.src.generate_ratings_intg import player_rt_intg
import pandas as pd

class PlayerRtIntgModel(BaseModel):
    """Model for the player ratings (integral method) page."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None

    def load_data(self) -> None:
        """Load player ratings data using the integral method."""
        try:
            # Call the existing player_rt_intg function
            self.data = player_rt_intg()
            self.error_message = None
        except Exception as e:
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_data(self) -> dict[str, Any]:
        """Get the current state of the model."""
        if self.data is None:
            self.load_data()
        
        return {
            "title": "Рейтинг игроков (интегральный метод)",
            "data": self.data,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """Update model data (not needed for this page)."""
        pass 