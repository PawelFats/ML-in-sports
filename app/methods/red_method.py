from .base import IRatingStrategy
from ..config import METRICS, P_METRICS
from ..utils import rename_columns
import pandas as pd
import streamlit as st

class RedMethodStrategy(IRatingStrategy):
    def calculate(self, df_compile: pd.DataFrame, df_history: pd.DataFrame = None) -> pd.DataFrame:
        df_rated = self._process_and_rate(df_compile)
        return self._postprocess(df_rated)

    def _process_and_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_stats = self._calc_stats(df)
        df_with_points = self._add_points(df_stats)
        return df_with_points

    def _calc_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[df['amplua'].isin([9, 10])]
              .groupby(['ID player', 'amplua'])
              .agg(
                  games=('ID game', 'nunique'),
                  goals=('goals', 'sum'),
                  assists=('assists', 'sum'),
                  throws_by=('throws by', 'sum'),
                  shot_on_target=('a shot on target', 'sum'),
                  blocked_throws=('blocked throws', 'sum'),
                  p_m=('p/m', 'sum')
              ).reset_index()
        )

    def _add_points(self, df: pd.DataFrame) -> pd.DataFrame:
        def calc_sub(df_sub: pd.DataFrame, coeff: float) -> pd.DataFrame:
            df_sub = df_sub.copy()
            for col in METRICS:
                df_sub[f'p_{col}'] = ((df_sub[col] + coeff * df_sub['games']) ** 2) / df_sub['games']
            df_sub['player_rating'] = df_sub[P_METRICS].sum(axis=1)
            return df_sub

        df_def = calc_sub(df[df['amplua'] == 9], coeff=2/3)
        df_for = calc_sub(df[df['amplua'] == 10], coeff=1/6)
        return pd.concat([df_def, df_for], ignore_index=True).round(2)

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return rename_columns(df)

    def calculate_season(self, df_compile: pd.DataFrame, df_history: pd.DataFrame,
                         season_id: int, player_ids: list[int] = None) -> pd.DataFrame:
        df_hist_season = df_history[df_history['ID season'] == season_id]
        df_season = df_compile.merge(
            df_hist_season[['ID']], left_on='ID game', right_on='ID', how='inner')
        if player_ids:
            df_season = df_season[df_season['ID player'].isin(player_ids)]

        st.write(f"Игры: {df_season['ID game'].nunique()}, Команды: {df_season['ID team'].nunique()}")
        counts = df_season.groupby('amplua')['ID player'].nunique()
        for amp, cnt in counts.items():
            st.write(f"Амплуа {amp}: {cnt} игроков")

        df_season_stats = self._process_and_rate(df_season)
        return self._postprocess(df_season_stats)