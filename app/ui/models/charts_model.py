import pandas as pd
from ui.core.base import BaseModel
from typing import Any, Optional, List
from pathlib import Path

class ChartsModel(BaseModel):
    """Model for the team ELO charts page."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None
        self.selected_team: Optional[int] = None
        self.data_path = Path("data/targeted/game_stats_one_r.csv")

    def load_data(self) -> None:
        """Load the game statistics data from CSV file."""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data["date"] = pd.to_datetime(self.data["date"])
            self.data = self.data.sort_values(by="date", ascending=True)
            self.error_message = None
        except Exception as e:
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_team_list(self) -> List[int]:
        """Get list of available teams."""
        if self.data is None:
            self.load_data()
        return [] if self.data is None else sorted(self.data["ID team"].unique())

    def get_team_data(self, team_id: int) -> Optional[pd.DataFrame]:
        """Get data for a specific team."""
        if self.data is None:
            self.load_data()
        return None if self.data is None else self.data[self.data["ID team"] == team_id]

    def get_data(self) -> dict[str, Any]:
        """Get the current state of the model."""
        if self.data is None:
            self.load_data()
        
        return {
            "title": "Графики статистики",
            "data": self.data,
            "teams": self.get_team_list(),
            "selected_team_data": self.get_team_data(self.selected_team) if self.selected_team else None,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """Update selected team."""
        if isinstance(data, dict) and "selected_team" in data:
            self.selected_team = data["selected_team"] 