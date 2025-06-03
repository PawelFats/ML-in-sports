import pandas as pd
from ui.core.base import BaseModel
from typing import Any, Optional
from pathlib import Path

class RankingsModel(BaseModel):
    """Model for the team rankings page."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None
        self.data_path = Path("data/targeted/team_ratings_merge.csv")

    def load_data(self) -> None:
        """Load the team rankings data from CSV file."""
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
            "title": "Рейтинги команд",
            "description": "Здесь будет информация о рейтингах команд.",
            "data": self.data,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """Update model data (not needed for rankings page)."""
        pass 