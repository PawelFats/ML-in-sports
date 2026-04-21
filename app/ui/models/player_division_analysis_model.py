from ui.core.base import BaseModel
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from src.generate_rating_red import process_seasons, load_data
from src.generate_rating_red import calculate_points, process_and_save


class PlayerDivisionAnalysisModel(BaseModel):
    """Модель для анализа рейтингов игроков по дивизионам."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.error_message: Optional[str] = None
    
    def get_available_seasons(self) -> list[int]:
        """Получает список доступных сезонов из того же источника, что и division_model."""
        try:
            # Используем тот же источник, что и division_model - game_stats_one_r.csv
            game_stats_path = Path('data/targeted/game_stats_one_r.csv')
            if game_stats_path.exists():
                game_stats = pd.read_csv(game_stats_path)
                available_seasons = sorted(game_stats['ID season'].dropna().unique().astype(int).tolist())
                return available_seasons
            else:
                # Fallback на game_history.csv, если основной файл не найден
                game_history_path = Path('data/raw/game_history.csv')
                if game_history_path.exists():
                    df_history = pd.read_csv(game_history_path)
                    available_seasons = sorted(df_history['ID season'].dropna().unique().astype(int).tolist())
                    return available_seasons
        except Exception as e:
            return []
        return []
    
    def get_player_ratings_by_season_division(
        self,
        season: int,
        coef_def: float = 2/3,
        coef_att: float = 1/6,
        metric_weights: Optional[Dict[str, float]] = None,
        division_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Получает рейтинги игроков для конкретного сезона с информацией о дивизионах.
        
        Returns:
            Словарь с данными о рейтингах игроков по дивизионам
        """
        try:
            # Загружаем данные
            df_history, df_compile_stats, _ = load_data()
            
            # Фильтруем по сезону
            df_hist_season = df_history[df_history["ID season"] == season]
            df_season = pd.merge(
                df_compile_stats,
                df_hist_season[["ID", "division", "ID season"]],
                left_on="ID game",
                right_on="ID",
                how="inner"
            )
            
            if df_season.empty:
                return {
                    'error': f'Нет данных для сезона {season}',
                    'player_ratings': pd.DataFrame(),
                    'division_stats': pd.DataFrame()
                }
            
            # Группируем по игрокам, амплуа и команде
            df_grouped = df_season.groupby(['ID player', 'amplua', 'ID team']).agg(
                games=('ID game', 'nunique'),
                goals=('goals', 'sum'),
                assists=('assists', 'sum'),
                assists_2=('assists_2', 'sum'),
                throws_by=('throws by', 'sum'),
                shot_on_target=('a shot on target', 'sum'),
                blocked_throws=('blocked throws', 'sum'),
                p_m=('p/m', 'sum')
            ).reset_index()
            
            # Используем стандартные веса, если не указаны
            if metric_weights is None:
                metric_weights = {
                    'goals': 1.0,
                    'assists': 1.0,
                    'assists_2': 1.0,
                    'throws_by': 1.0,
                    'shot_on_target': 1.0,
                    'blocked_throws': 1.0,
                    'p/m': 1.0
                }
            
            # Вычисляем рейтинги для защитников и нападающих
            df_def = calculate_points(df_grouped, coef_def, 9, metric_weights)
            df_att = calculate_points(df_grouped, coef_att, 10, metric_weights)
            df_players = pd.concat([df_def, df_att], ignore_index=True)
            
            # Получаем информацию о дивизионах команд
            df_team_division = df_season.groupby('ID team', as_index=False).agg({
                'division': 'first',
                'ID season': 'first'
            })
            
            # Объединяем рейтинги игроков с информацией о дивизионах
            df_players_with_division = pd.merge(
                df_players,
                df_team_division[['ID team', 'division']],
                on='ID team',
                how='left'
            )
            
            # Применяем веса дивизионов, если указаны
            if division_weights:
                df_players_with_division['division'] = df_players_with_division['division'].astype('Int64')
                df_players_with_division['player_rating'] = df_players_with_division.apply(
                    lambda r: r['player_rating'] * float(division_weights.get(int(r['division']) if pd.notna(r['division']) else 0, 1.0)),
                    axis=1
                )
            
            # Вычисляем статистику по дивизионам
            division_stats = []
            for div in df_players_with_division['division'].dropna().unique():
                div_players = df_players_with_division[df_players_with_division['division'] == div]
                if len(div_players) > 0:
                    avg_rating = div_players['player_rating'].mean()
                    std_rating = div_players['player_rating'].std()
                    median_rating = div_players['player_rating'].median()
                    min_rating = div_players['player_rating'].min()
                    max_rating = div_players['player_rating'].max()
                    num_players = len(div_players)
                    
                    # Игроки, которые сильно отклоняются от среднего (более 1.5 стандартных отклонений)
                    threshold_low = avg_rating - 1.5 * std_rating
                    threshold_high = avg_rating + 1.5 * std_rating
                    
                    outliers_low = div_players[div_players['player_rating'] < threshold_low]
                    outliers_high = div_players[div_players['player_rating'] > threshold_high]
                    
                    division_stats.append({
                        'division': int(div),
                        'avg_rating': avg_rating,
                        'std_rating': std_rating,
                        'median_rating': median_rating,
                        'min_rating': min_rating,
                        'max_rating': max_rating,
                        'num_players': num_players,
                        'outliers_low_count': len(outliers_low),
                        'outliers_high_count': len(outliers_high),
                        'outliers_total': len(outliers_low) + len(outliers_high),
                        'threshold_low': threshold_low,
                        'threshold_high': threshold_high
                    })
            
            division_stats_df = pd.DataFrame(division_stats)
            
            return {
                'error': None,
                'player_ratings': df_players_with_division,
                'division_stats': division_stats_df,
                'season': season
            }
            
        except Exception as e:
            return {
                'error': f'Ошибка при получении рейтингов: {str(e)}',
                'player_ratings': pd.DataFrame(),
                'division_stats': pd.DataFrame()
            }
    
    def get_data(self) -> Dict[str, Any]:
        """Возвращает текущие данные модели."""
        return {
            "title": "Анализ рейтингов игроков по дивизионам",
            "data": self.data,
            "error_message": self.error_message
        }
    
    def update(self, data: Any) -> None:
        """Метод обновления модели."""
        pass

