from ui.core.base import BaseController
from ui.models.player_division_analysis_model import PlayerDivisionAnalysisModel
from ui.views.player_division_analysis_view import PlayerDivisionAnalysisView
from typing import Dict, Any


class PlayerDivisionAnalysisController(BaseController):
    """Контроллер для страницы анализа рейтингов игроков по дивизионам."""
    
    def __init__(self):
        model = PlayerDivisionAnalysisModel()
        view = PlayerDivisionAnalysisView()
        super().__init__(model, view)
    
    def initialize(self) -> None:
        """Инициализация страницы анализа рейтингов игроков по дивизионам."""
        data = self.model.get_data()
        self.view.update(data)
        self.view.render()
    
    def handle_input(self, input_data: Dict[str, Any]) -> None:
        """Обработка ввода пользователя."""
        # Для этой страницы обработка ввода происходит напрямую в представлении
        pass

