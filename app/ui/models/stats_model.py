import pandas as pd
from ui.core.base import BaseModel
from typing import Any, Optional
from pathlib import Path

class StatsModel(BaseModel):
    """Model for the stats page that handles game statistics data."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None
        self.data_path = Path("data/targeted/game_stats_one_r.csv")

    def load_data(self) -> None:
        """Load the game statistics data from CSV file."""
        try:
            self.data = pd.read_csv(self.data_path)
            self.error_message = None
        except Exception as e:
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_data(self) -> dict[str, Any]:
        """Get the current state of the model."""
        if self.data is None:
            self.load_data()
        
        return {
            "title": "Статистика команд",
            "data": self.data,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """Update model data (not needed for stats page)."""
        pass 