import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

# Вставляем его в начало sys.path, чтобы все импорты шли от этого корня
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Импортируем основной класс приложения и контроллеры для разных разделов UI
from ui.core.app import HockeyAnalyticsApp
from ui.controllers.home_controller import HomeController
from ui.controllers.stats_controller import StatsController
from ui.controllers.rankings_controller import RankingsController
from ui.controllers.charts_controller import ChartsController
from ui.controllers.player_rt_intg_controller import PlayerRtIntgController
from ui.controllers.player_rt_red_controller import PlayerRtRedController
from ui.controllers.bayesian_controller import BayesianController

def main():
    """
    Основная функция: точка входа в приложение.
    Здесь создаётся экземпляр приложения, регистрируются контроллеры и запускается UI.
    """
    # Создаём приложение для аналитики хоккея
    app = HockeyAnalyticsApp()
    
    # Подключаем контроллеры: каждому даём название вкладки и экземпляр контроллера
    app.register_controller("Главная", HomeController())
    app.register_controller("Таблица с данными за каждую игру", StatsController())
    app.register_controller("Актуальные рейтинги команд", RankingsController())
    app.register_controller("ELO рейтинг команды за все время", ChartsController())
    app.register_controller("Рейтинг игроков (интегральный метод)", PlayerRtIntgController())
    app.register_controller("Рейтинг игроков (советский метод)", PlayerRtRedController())
    app.register_controller("Байесовский метод", BayesianController())
    
    # Запускаем цикл обработки событий и отображение интерфейса
    app.run()

# Если файл запускается как главный модуль, вызываем main()
if __name__ == "__main__":
    main()
