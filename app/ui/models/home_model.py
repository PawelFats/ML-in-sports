from ui.core.base import BaseModel
from typing import Dict, Any

class HomeModel(BaseModel):
    """Model for the home page."""
    
    def __init__(self):
        self.welcome_data = {
            "title": "Добро пожаловать!",
            "description": "Это приложение для анализа хоккейной статистики.",
            "instruction": "Выберите раздел в боковом меню для продолжения."
        }

    def get_data(self) -> Dict[str, str]:
        """Get the welcome data."""
        return self.welcome_data

    def update(self, data: Any) -> None:
        """Update model data (not needed for home page)."""
        pass 