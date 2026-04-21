import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
from src.models import (
    process_season_data,
    scale_and_select_features,
    simulate_all_matches,
    calculate_average_highest_probabilities,
    rank_teams,
    get_team_win_probability,
    normalize_probabilities
)

class DivisionModel:
    """Модель для работы с распределением команд по дивизионам."""
    
    def __init__(self):
        self.game_stats_path = Path('data/targeted/game_stats_one_r.csv')
        self.model_path = Path('MODEL.pkl')
    
    def get_available_seasons(self) -> list[int]:
        """
        Получает список доступных сезонов из данных.
        
        Returns:
            Отсортированный список номеров сезонов
        """
        try:
            game_stats = pd.read_csv(self.game_stats_path)
            available_seasons = sorted(game_stats['ID season'].dropna().unique().astype(int).tolist())
            return available_seasons
        except Exception as e:
            return []
    
    def _filter_divisions_with_sufficient_games(
        self, 
        division_teams: Dict[Any, list], 
        game_before_season: pd.DataFrame,
        min_games_per_team: int = 5
    ) -> Dict[Any, list]:
        """
        Фильтрует дивизионы, для которых недостаточно игр в истории.
        
        Args:
            division_teams: словарь {division: [team_ids]}
            game_before_season: DataFrame с историей игр до сезона
            min_games_per_team: минимальное количество игр на команду в дивизионе
            
        Returns:
            Отфильтрованный словарь дивизионов
        """
        filtered_divisions = {}
        
        for division, teams in division_teams.items():
            if not teams or len(teams) == 0:
                continue
            
            # Проверяем количество игр для команд в этом дивизионе
            division_team_ids = set(teams)
            division_games = game_before_season[
                (game_before_season['ID team'].isin(division_team_ids)) |
                (game_before_season['ID opponent'].isin(division_team_ids))
            ]
            
            # Проверяем, что есть достаточно игр для каждой команды
            teams_with_sufficient_games = []
            for team_id in teams:
                team_games = division_games[
                    (division_games['ID team'] == team_id) |
                    (division_games['ID opponent'] == team_id)
                ]
                if len(team_games) >= min_games_per_team:
                    teams_with_sufficient_games.append(team_id)
            
            # Если в дивизионе осталось достаточно команд с достаточным количеством игр
            if len(teams_with_sufficient_games) >= 2:  # Минимум 2 команды для матчей
                filtered_divisions[division] = teams_with_sufficient_games
        
        return filtered_divisions
    
    def evaluate_league_distribution(self, season: int) -> Dict[str, Any]:
        """
        Оценивает текущее распределение команд по дивизионам в лиге.
        
        Args:
            season: номер сезона для оценки
            
        Returns:
            Словарь с результатами оценки
        """
        try:
            # Получаем данные сезона
            game_before_season, unique_teams = process_season_data(
                season, 
                game_stats_path=str(self.game_stats_path)
            )
            
            # Проверяем наличие достаточного количества игр перед масштабированием
            if game_before_season.empty or len(game_before_season) < 1:
                return {
                    'error': 'Недостаточно игровой истории для оценки дивизионов',
                    'division_probabilities': pd.Series(dtype=float),
                    'division_teams': {},
                    'match_results': pd.DataFrame()
                }
            
            # Масштабируем данные
            game_before_season_scaled = scale_and_select_features(game_before_season, min_games=1)
            game_before_season_scaled["result"] = game_before_season_scaled["result"].map({'W': 1, 'L': -1, 'D': 0})
            
            # Формируем словарь дивизионов
            division_teams = {}
            for _, row in unique_teams.iterrows():
                division = row['division']
                team_id = row['ID team']
                
                # Пропускаем NaN дивизионы
                if pd.isna(division):
                    continue
                    
                if division not in division_teams:
                    division_teams[division] = []
                division_teams[division].append(team_id)
            
            # Фильтруем только дивизионы с командами
            division_teams = {k: v for k, v in division_teams.items() if v and len(v) > 0}
            
            # Фильтруем дивизионы с недостаточным количеством игр
            division_teams = self._filter_divisions_with_sufficient_games(
                division_teams, 
                game_before_season,
                min_games_per_team=5
            )
            
            if not division_teams:
                return {
                    'error': 'Нет дивизионов с достаточным количеством игр для оценки',
                    'division_probabilities': pd.Series(dtype=float),
                    'division_teams': {},
                    'match_results': pd.DataFrame()
                }
            
            # Симулируем матчи
            match_results = simulate_all_matches(
                division_teams=division_teams,
                model=None,
                matches=game_before_season_scaled,
                model_path=str(self.model_path)
            )
            
            # Вычисляем средние значения наибольших вероятностей
            division_probabilities = calculate_average_highest_probabilities(match_results)
            
            return {
                'error': None,
                'division_probabilities': division_probabilities,
                'division_teams': division_teams,
                'match_results': match_results
            }
            
        except Exception as e:
            return {
                'error': f'Ошибка: {str(e)}',
                'division_probabilities': pd.Series(dtype=float),
                'division_teams': {},
                'match_results': pd.DataFrame()
            }
    
    def rank_teams_into_divisions(
        self, 
        season: int, 
        num_divisions: int, 
        min_teams_per_division: int,
        num_generations: int = 100,
        population_size: int = 300,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Распределяет команды по дивизионам с помощью генетического алгоритма.
        
        Args:
            season: номер сезона
            num_divisions: количество дивизионов
            min_teams_per_division: минимальное количество команд в дивизионе
            num_generations: количество поколений генетического алгоритма
            population_size: размер популяции
            
        Returns:
            Словарь с результатами распределения
        """
        try:
            if progress_callback:
                progress_callback("Загрузка данных сезона...", 0.1)
            
            # Получаем данные сезона
            game_before_season, unique_teams = process_season_data(
                season,
                game_stats_path=str(self.game_stats_path)
            )
            
            # Проверяем наличие достаточного количества игр перед масштабированием
            if game_before_season.empty or len(game_before_season) < 1:
                return {
                    'error': 'Недостаточно игровой истории для распределения команд',
                    'ranked_teams_df': pd.DataFrame(),
                    'division_teams': {},
                    'division_probabilities': pd.Series(dtype=float),
                    'match_results': pd.DataFrame()
                }
            
            if progress_callback:
                progress_callback(f"Запуск генетического алгоритма ({num_generations} поколений)...", 0.2)
            
            # Распределяем команды
            ranked_teams_df = rank_teams(
                list_team=unique_teams,
                model_file=str(self.model_path),
                num_divisions=num_divisions,
                min_teams_per_division=min_teams_per_division,
                num_generations=num_generations,
                population_size=population_size,
                matches=None,
                game_stats_path=str(self.game_stats_path)
            )
            
            if progress_callback:
                progress_callback("Формирование дивизионов...", 0.6)
            
            # Формируем словарь дивизионов
            division_teams = {}
            for _, row in ranked_teams_df.iterrows():
                division = row['division']
                team_id = row['ID team']
                
                if division not in division_teams:
                    division_teams[division] = []
                division_teams[division].append(team_id)
            
            # Фильтруем только дивизионы с командами
            division_teams = {k: v for k, v in division_teams.items() if v and len(v) > 0}
            
            # Фильтруем дивизионы с недостаточным количеством игр
            division_teams = self._filter_divisions_with_sufficient_games(
                division_teams, 
                game_before_season,
                min_games_per_team=5
            )
            
            if not division_teams:
                return {
                    'error': 'Нет дивизионов с достаточным количеством игр для оценки',
                    'ranked_teams_df': ranked_teams_df,
                    'division_teams': {},
                    'division_probabilities': pd.Series(dtype=float),
                    'match_results': pd.DataFrame()
                }
            
            if progress_callback:
                progress_callback("Масштабирование данных...", 0.7)
            
            # Масштабируем данные для симуляции
            game_before_season_scaled = scale_and_select_features(game_before_season, min_games=1)
            game_before_season_scaled["result"] = game_before_season_scaled["result"].map({'W': 1, 'L': -1, 'D': 0})
            
            if progress_callback:
                progress_callback("Симуляция матчей между командами...", 0.85)
            
            # Симулируем матчи
            match_results = simulate_all_matches(
                division_teams=division_teams,
                model=None,
                matches=game_before_season_scaled,
                model_path=str(self.model_path)
            )
            
            if progress_callback:
                progress_callback("Вычисление вероятностей...", 0.95)
            
            # Вычисляем средние значения наибольших вероятностей
            division_probabilities = calculate_average_highest_probabilities(match_results)
            
            if progress_callback:
                progress_callback("Завершено!", 1.0)
            
            return {
                'error': None,
                'ranked_teams_df': ranked_teams_df,
                'division_teams': division_teams,
                'division_probabilities': division_probabilities,
                'match_results': match_results
            }
            
        except Exception as e:
            return {
                'error': f'Ошибка: {str(e)}',
                'ranked_teams_df': pd.DataFrame(),
                'division_teams': {},
                'division_probabilities': pd.Series(dtype=float),
                'match_results': pd.DataFrame()
            }
    
    def evaluate_custom_distribution(
        self, 
        team_rangirov_df: pd.DataFrame, 
        season: int
    ) -> Dict[str, Any]:
        """
        Оценивает пользовательское распределение команд по дивизионам.
        
        Args:
            team_rangirov_df: DataFrame с колонками 'ID team' и 'division'
            season: номер сезона
            
        Returns:
            Словарь с результатами оценки
        """
        try:
            # Получаем данные сезона
            game_before_season, unique_teams = process_season_data(
                season,
                game_stats_path=str(self.game_stats_path)
            )
            
            # Проверяем наличие достаточного количества игр перед масштабированием
            if game_before_season.empty or len(game_before_season) < 1:
                return {
                    'error': 'Недостаточно игровой истории для оценки распределения',
                    'division_probabilities': pd.Series(dtype=float),
                    'division_teams': {},
                    'match_results': pd.DataFrame()
                }
            
            # Формируем словарь дивизионов из пользовательских данных
            division_teams = {}
            for _, row in team_rangirov_df.iterrows():
                division = row['division']
                team_id = row['ID team']
                
                # Пропускаем NaN дивизионы
                if pd.isna(division):
                    continue
                    
                if division not in division_teams:
                    division_teams[division] = []
                division_teams[division].append(team_id)
            
            # Фильтруем только дивизионы с командами
            division_teams = {k: v for k, v in division_teams.items() if v and len(v) > 0}
            
            # Фильтруем дивизионы с недостаточным количеством игр
            division_teams = self._filter_divisions_with_sufficient_games(
                division_teams, 
                game_before_season,
                min_games_per_team=5
            )
            
            if not division_teams:
                return {
                    'error': 'Нет дивизионов с достаточным количеством игр для оценки',
                    'division_probabilities': pd.Series(dtype=float),
                    'division_teams': {},
                    'match_results': pd.DataFrame()
                }
            
            # Масштабируем данные
            game_before_season_scaled = scale_and_select_features(game_before_season, min_games=1)
            game_before_season_scaled["result"] = game_before_season_scaled["result"].map({'W': 1, 'L': -1, 'D': 0})
            
            # Симулируем матчи
            match_results = simulate_all_matches(
                division_teams=division_teams,
                model=None,
                matches=game_before_season_scaled,
                model_path=str(self.model_path)
            )
            
            # Вычисляем средние значения наибольших вероятностей
            division_probabilities = calculate_average_highest_probabilities(match_results)
            
            return {
                'error': None,
                'division_probabilities': division_probabilities,
                'division_teams': division_teams,
                'match_results': match_results
            }
            
        except Exception as e:
            return {
                'error': f'Ошибка: {str(e)}',
                'division_probabilities': pd.Series(dtype=float),
                'division_teams': {},
                'match_results': pd.DataFrame()
            }
    
    def predict_season_matches(
        self,
        season: int,
        division_teams: Dict[Any, list],
        game_before_season: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Прогнозирует результаты всех матчей сезона с заданным распределением команд.
        
        Args:
            season: номер сезона
            division_teams: словарь {division: [team_ids]} - распределение команд
            game_before_season: DataFrame с историей игр до сезона
            
        Returns:
            DataFrame с прогнозами матчей
        """
        try:
            # Загружаем модель
            model = joblib.load(self.model_path)
            
            # Масштабируем данные
            game_before_season_scaled = scale_and_select_features(game_before_season, min_games=1)
            game_before_season_scaled["result"] = game_before_season_scaled["result"].map({'W': 1, 'L': -1, 'D': 0})
            
            # Загружаем реальные матчи сезона
            game_stats = pd.read_csv(self.game_stats_path)
            season_games = game_stats[game_stats['ID season'] == season].copy()
            
            predictions = []
            
            for _, game in season_games.iterrows():
                team1_id = game['ID team']
                team2_id = game['ID opponent']
                real_result = game['result']
                
                # Пропускаем если команды не валидны
                if pd.isna(team1_id) or pd.isna(team2_id):
                    continue
                
                # Проверяем, что обе команды в одном дивизионе
                team1_division = None
                team2_division = None
                
                for div, teams in division_teams.items():
                    if team1_id in teams:
                        team1_division = div
                    if team2_id in teams:
                        team2_division = div
                
                # Пропускаем матчи между командами из разных дивизионов
                if team1_division != team2_division or team1_division is None:
                    continue
                
                # Получаем вероятности
                win_prob1 = get_team_win_probability(model, game_before_season_scaled, team1_id, team2_id)
                win_prob2 = get_team_win_probability(model, game_before_season_scaled, team2_id, team1_id)
                
                if not np.isfinite(win_prob1) or not np.isfinite(win_prob2):
                    continue
                
                # Нормализуем вероятности
                probabilities = normalize_probabilities([win_prob1, win_prob2])
                
                # Определяем прогноз (W если вероятность > 0.5, иначе L)
                predicted_result = 'W' if probabilities[0] > 0.5 else 'L'
                
                predictions.append({
                    'ID game': game['ID game'],
                    'ID team': team1_id,
                    'ID opponent': team2_id,
                    'division': team1_division,
                    'predicted_prob_team': probabilities[0],
                    'predicted_prob_opponent': probabilities[1],
                    'predicted_result': predicted_result,
                    'real_result': real_result,
                    'correct': 1 if predicted_result == real_result else 0
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            return pd.DataFrame()
    
    def validate_distribution(
        self,
        season: int,
        division_teams: Dict[Any, list]
    ) -> Dict[str, Any]:
        """
        Валидирует распределение команд, сравнивая прогнозы с реальными результатами.
        
        Args:
            season: номер сезона для валидации
            division_teams: словарь {division: [team_ids]} - распределение команд
            
        Returns:
            Словарь с метриками валидации
        """
        try:
            # Получаем данные до сезона
            game_before_season, _ = process_season_data(
                season,
                game_stats_path=str(self.game_stats_path)
            )
            
            if game_before_season.empty or len(game_before_season) < 1:
                return {
                    'error': 'Недостаточно игровой истории для валидации',
                    'accuracy': 0,
                    'total_matches': 0,
                    'correct_predictions': 0
                }
            
            # Получаем прогнозы
            predictions_df = self.predict_season_matches(
                season=season,
                division_teams=division_teams,
                game_before_season=game_before_season
            )
            
            if predictions_df.empty:
                return {
                    'error': 'Не удалось получить прогнозы',
                    'accuracy': 0,
                    'total_matches': 0,
                    'correct_predictions': 0
                }
            
            # Вычисляем метрики
            total_matches = len(predictions_df)
            correct_predictions = predictions_df['correct'].sum()
            accuracy = correct_predictions / total_matches if total_matches > 0 else 0
            
            # Brier Score (для вероятностных прогнозов)
            predictions_df['brier_score'] = predictions_df.apply(
                lambda row: (row['predicted_prob_team'] - (1 if row['real_result'] == 'W' else 0))**2,
                axis=1
            )
            brier_score = predictions_df['brier_score'].mean()
            
            # Распределение по дивизионам
            division_stats = predictions_df.groupby('division').agg({
                'correct': ['sum', 'count'],
                'brier_score': 'mean'
            }).reset_index()
            division_stats.columns = ['division', 'correct', 'total', 'brier_score']
            division_stats['accuracy'] = division_stats['correct'] / division_stats['total']
            
            return {
                'error': None,
                'accuracy': accuracy,
                'total_matches': total_matches,
                'correct_predictions': correct_predictions,
                'brier_score': brier_score,
                'predictions_df': predictions_df,
                'division_stats': division_stats
            }
            
        except Exception as e:
            return {
                'error': f'Ошибка валидации: {str(e)}',
                'accuracy': 0,
                'total_matches': 0,
                'correct_predictions': 0
            }
    
    def compare_distributions(
        self,
        season: int
    ) -> Dict[str, Any]:
        """
        Сравнивает эффективность текущего распределения лиги и автоматического распределения.
        
        Args:
            season: номер сезона для сравнения
            
        Returns:
            Словарь с результатами сравнения
        """
        try:
            # Получаем текущее распределение лиги
            game_before_season, unique_teams = process_season_data(
                season,
                game_stats_path=str(self.game_stats_path)
            )
            
            # Формируем словарь дивизионов текущего распределения
            league_division_teams = {}
            for _, row in unique_teams.iterrows():
                division = row['division']
                team_id = row['ID team']
                
                if pd.isna(division):
                    continue
                    
                if division not in league_division_teams:
                    league_division_teams[division] = []
                league_division_teams[division].append(team_id)
            
            league_division_teams = {k: v for k, v in league_division_teams.items() if v and len(v) > 0}
            
            # Фильтруем дивизионы с недостаточным количеством игр
            league_division_teams = self._filter_divisions_with_sufficient_games(
                league_division_teams,
                game_before_season,
                min_games_per_team=5
            )
            
            # Валидируем текущее распределение
            league_validation = self.validate_distribution(season, league_division_teams)
            
            # Получаем автоматическое распределение
            num_divisions = len(league_division_teams)
            if num_divisions < 2:
                num_divisions = 4
            
            min_teams = min([len(teams) for teams in league_division_teams.values()]) if league_division_teams else 3
            if min_teams < 2:
                min_teams = 3
            
            auto_result = self.rank_teams_into_divisions(
                season=season,
                num_divisions=num_divisions,
                min_teams_per_division=min_teams,
                num_generations=100,
                progress_callback=None
            )
            
            if auto_result.get('error'):
                return {
                    'error': f'Не удалось получить автоматическое распределение: {auto_result.get("error")}',
                    'league_validation': league_validation,
                    'auto_validation': None
                }
            
            auto_division_teams = auto_result.get('division_teams', {})
            
            # Фильтруем дивизионы с недостаточным количеством игр
            auto_division_teams = self._filter_divisions_with_sufficient_games(
                auto_division_teams,
                game_before_season,
                min_games_per_team=5
            )
            
            # Валидируем автоматическое распределение
            auto_validation = self.validate_distribution(season, auto_division_teams)
            
            return {
                'error': None,
                'league_validation': league_validation,
                'auto_validation': auto_validation,
                'league_division_teams': league_division_teams,
                'auto_division_teams': auto_division_teams
            }
            
        except Exception as e:
            return {
                'error': f'Ошибка сравнения: {str(e)}',
                'league_validation': None,
                'auto_validation': None
            }

