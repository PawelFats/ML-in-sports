import streamlit as st
from ..data_loader import DataLoader
from ..factory import StrategyFactory
from ..calculators.stats_calculator import StatsCalculator
from ..plotters.player_plotter import PlayerPlotter

class UIController:
    def __init__(self):
        self.loader = DataLoader()

    def run(self):
        st.title("Анализ игроков — Красный метод")
        df_hist = self.loader.load_history()
        df_comp = self.loader.load_compile_stats()

        action = st.sidebar.selectbox("Действие", [
            "Актуальный рейтинг игроков",
            "Статистика за сезон"
        ])

        if action == "Актуальный рейтинг игроков":
            strat = StrategyFactory.get("red")
            df_r = StatsCalculator(strat).compute(df_comp)
            st.dataframe(df_r)

        else:  # Статистика за сезон
            season = st.sidebar.selectbox("Сезон", sorted(df_hist['ID season'].unique()))
            strat = StrategyFactory.get("red")
            df_r = strat.calculate_season(df_comp, df_hist, season)
            plotter = PlayerPlotter()
            for fig in plotter.plot(df_r, season):
                st.pyplot(fig)
            st.dataframe(df_r)