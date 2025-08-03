import os
import sys

# Определяем корневую папку проекта и добавляем её в список путей Python,
# чтобы можно было импортировать модули из корня без относительных импортов
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')  # Находим папку на уровень выше текущего файла
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Импортируем основной класс приложения и контроллеры для разных разделов UI
from app.ui.core.app import HockeyAnalyticsApp
from app.ui.controllers.home_controller import HomeController
from app.ui.controllers.stats_controller import StatsController
from app.ui.controllers.rankings_controller import RankingsController
from app.ui.controllers.charts_controller import ChartsController
from app.ui.controllers.player_rt_intg_controller import PlayerRtIntgController
from app.ui.controllers.player_rt_red_controller import PlayerRtRedController
from app.ui.controllers.bayesian_controller import BayesianController

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
