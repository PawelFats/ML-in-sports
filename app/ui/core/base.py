from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModel(ABC):
    """Base class for all models in the application."""
    
    @abstractmethod
    def get_data(self) -> Any:
        """Retrieve data from the model."""
        pass

    @abstractmethod
    def update(self, data: Any) -> None:
        """Update model data."""
        pass

class BaseView(ABC):
    """Base class for all views in the application."""
    
    @abstractmethod
    def render(self) -> None:
        """Render the view."""
        pass

    @abstractmethod
    def update(self, data: Any) -> None:
        """Update the view with new data."""
        pass

class BaseController(ABC):
    """Base class for all controllers in the application."""
    
    def __init__(self, model: BaseModel, view: BaseView):
        self.model = model
        self.view = view

    @abstractmethod
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Handle user input and update model/view accordingly."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the controller, model, and view."""
        pass 