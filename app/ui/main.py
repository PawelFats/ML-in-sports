import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.core.app import HockeyAnalyticsApp
from ui.controllers.home_controller import HomeController
from ui.controllers.stats_controller import StatsController
from ui.controllers.rankings_controller import RankingsController
from ui.controllers.charts_controller import ChartsController
from ui.controllers.player_rt_intg_controller import PlayerRtIntgController
from ui.controllers.player_rt_red_controller import PlayerRtRedController
from ui.controllers.bayesian_controller import BayesianController

def main():
    """Main entry point of the application."""
    app = HockeyAnalyticsApp()
    
    # Register all controllers
    app.register_controller("Главная", HomeController())
    app.register_controller("Таблица с данными за каждую игру", StatsController())
    app.register_controller("Актуальные рейтинги команд", RankingsController())
    app.register_controller("ELO рейтинг команды за все время", ChartsController())
    app.register_controller("Рейтинг игроков (интегральный метод)", PlayerRtIntgController())
    app.register_controller("Рейтинг игроков (советский метод)", PlayerRtRedController())
    app.register_controller("Байесовский метод", BayesianController())
    
    # Run the application
    app.run()

if __name__ == "__main__":
    main()
