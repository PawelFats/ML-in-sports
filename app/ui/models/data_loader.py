
import requests
import pandas as pd
import warnings

from pathlib import Path
from requests.packages.urllib3.exceptions import InsecureRequestWarning

class DataLoader:
    """
    Загрузчик «сырых» данных: скачивает CSV-файлы по ссылкам и выполняет базовую предобработку.
    """
    BASE_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

    @classmethod
    def fetch_players(cls) -> pd.DataFrame:
        url = (
            'https://llhn.ru/export?f=csv&request=players'
            '&login=export_csv&pass=a3cd9b8FB2ac'
        )
        warnings.simplefilter('ignore', InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status()

        cls.BASE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = cls.BASE_RAW_DIR / 'players_amplua.csv'
        file_path.write_bytes(response.content)
        
        # Читаем CSV с разделителем ';'
        df = pd.read_csv(file_path, sep=';')

        # Словарь переименования
        rename_map = {
            'id': 'ID player',
            'skill': 'skill',
            'amplua': 'amplua'
        }
        # Переименовываем и выбираем только нужные столбцы
        df = df.rename(columns=rename_map)
        df = df[list(rename_map.values())]

        df.to_csv(file_path, index=False)
        return df

    @classmethod
    def fetch_player_stats(cls) -> pd.DataFrame:
        url = (
            'https://llhn.ru/export?f=csv&request=player_stats'
            '&login=export_csv&pass=a3cd9b8FB2ac'
        )
        warnings.simplefilter('ignore', InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status()

        cls.BASE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = cls.BASE_RAW_DIR / 'player_stats.csv'
        file_path.write_bytes(response.content)

        # Читаем CSV с разделителем ';'
        df = pd.read_csv(file_path, sep=';')

        # Словарь переименования
        rename_map = {
            'id': 'ID',
            'game': 'ID game',
            'player': 'ID player',
            'team': 'ID team',
            'time_on_ice': 'total time on ice',
            'kick_miss': 'throws by',
            'kick_target': 'a shot on target',
            'accurate_transmission': 'accurate transmission',
            'blocked_shots': 'blocked throws'
        }
        # Переименовываем и выбираем только нужные столбцы
        df = df.rename(columns=rename_map)
        df = df[list(rename_map.values())]

        df.to_csv(file_path, index=False)
        return df

    @classmethod
    def fetch_games_history(cls) -> pd.DataFrame:
        url = (
            'https://llhn.ru/export?f=csv&request=games'
            '&login=export_csv&pass=a3cd9b8FB2ac'
        )
        warnings.simplefilter('ignore', InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status()

        cls.BASE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = cls.BASE_RAW_DIR / 'game_history.csv'
        file_path.write_bytes(response.content)

        df = pd.read_csv(file_path, sep=';', engine='python')
        df.columns = df.columns.str.strip()
        rename_map = {
            'id': 'ID',
            'tournament': 'ID season',
            'first_team': 'ID firstTeam',
            'second_team': 'ID secondTeam',
            'gameid': 'ID game',
            'stageid': 'stage',
            'group_num': 'division',
            'game_date': 'date'
        }
        df = df.rename(columns=rename_map)
        selected_cols = [new for new in rename_map.values() if new in df.columns]
        df = df[selected_cols]

        df.to_csv(file_path, index=False)
        return df
        
    @classmethod
    def fetch_game_goals_and_passes(cls) -> pd.DataFrame:
        """
        Скачивает CSV с событиями игр (голы и передачи), сохраняет в data/raw,
        переименовывает указанные колонки.

        Returns:
            Обработанный DataFrame
        """
        url = (
            'https://llhn.ru/export?f=csv&request=game_events'
            '&login=export_csv&pass=a3cd9b8FB2ac'
        )
        warnings.simplefilter('ignore', InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status()

        cls.BASE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = cls.BASE_RAW_DIR / 'goals_and_passes.csv'
        file_path.write_bytes(response.content)

        # Читаем CSV (разделитель ';')
        df = pd.read_csv(file_path, sep=';', engine='python')
        df.columns = df.columns.str.strip()

        # Переименование колонок
        rename_map = {
            'id': 'ID',
            'game': 'ID game',
            'team': 'ID team',
            'goal': 'ID player scored',
            'assist': 'ID player assist',
            'assist2': 'ID player assist 2'
        }
        df = df.rename(columns=rename_map)

        # Сохраняем обратно
        df.to_csv(file_path, index=False)
        return df

    @classmethod
    def fetch_keeper_events(cls) -> pd.DataFrame:
        """
        Скачивает CSV с событиями вратарей, сохраняет в data/raw,
        переименовывает указанные колонки.

        Returns:
            Обработанный DataFrame
        """
        url = (
            'https://llhn.ru/export?f=csv&request=keeper_events'
            '&login=export_csv&pass=a3cd9b8FB2ac'
        )
        warnings.simplefilter('ignore', InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status()

        cls.BASE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = cls.BASE_RAW_DIR / 'goalkeeper_event.csv'
        file_path.write_bytes(response.content)

        # Читаем CSV (разделитель ';')
        df = pd.read_csv(file_path, sep=';', engine='python')
        df.columns = df.columns.str.strip()

        # Переименование колонок
        rename_map = {
            'id': 'ID',
            'game': 'ID game',
            'team': 'ID team',
            'player_in': 'IDp_in_ice',
            'player_out': 'IDp_out_ice',
            'event_time': 'timer'
        }
        df = df.rename(columns=rename_map)

        # Сохраняем обратно
        df.to_csv(file_path, index=False)
        return df

    @classmethod
    def fetch_game_plus_minus(cls) -> pd.DataFrame:
        """
        Скачивает CSV с показателями «+/-» игроков, сохраняет в data/raw,
        переименовывает указанные колонки.

        Returns:
            Обработанный DataFrame
        """
        url = (
            'https://llhn.ru/export?f=csv&request=game_event_players'
            '&login=export_csv&pass=a3cd9b8FB2ac'
        )
        warnings.simplefilter('ignore', InsecureRequestWarning)
        response = requests.get(url, verify=False)
        response.raise_for_status()

        cls.BASE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = cls.BASE_RAW_DIR / 'game_plus_minus.csv'
        file_path.write_bytes(response.content)

        # Читаем CSV (разделитель ';')
        df = pd.read_csv(file_path, sep=';', engine='python')
        df.columns = df.columns.str.strip()

        # Переименование колонок
        rename_map = {
            'id': 'ID',
            'game_event': 'ID event',
            'game': 'ID game',
            'team': 'ID team',
            'tournament': 'ID season',
            'goal_team': 'the scoring team',
            'player': 'ID player'
        }
        df = df.rename(columns=rename_map)

        # Сохраняем обратно
        df.to_csv(file_path, index=False)
        return df
