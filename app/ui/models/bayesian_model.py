from ui.core.base import BaseModel
from typing import Any, Optional
from app.src.method_bayesian import bayesian_analysis
import pandas as pd

class BayesianModel(BaseModel):
    """Model for the Bayesian analysis page."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None

    def load_data(self) -> None:
        """Load Bayesian analysis data."""
        try:
            # Call the existing bayesian_analysis function
            self.data = bayesian_analysis()
            self.error_message = None
        except Exception as e:
            self.data = None
            self.error_message = f"Ошибка загрузки данных: {str(e)}"

    def get_data(self) -> dict[str, Any]:
        """Get the current state of the model."""
        if self.data is None:
            self.load_data()
        
        return {
            "title": "Байесовский метод",
            "data": self.data,
            "error_message": self.error_message
        }

    def update(self, data: Any) -> None:
        """Update model data (not needed for this page)."""
        pass 