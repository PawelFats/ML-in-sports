# ml_in_sports/data_loader.py
import pandas as pd
import streamlit as st
from .config import DATA_PATH, CACHE_TTL

class DataLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path

    @st.cache_data(ttl=CACHE_TTL)
    def load_history(_self) -> pd.DataFrame:
        return pd.read_csv(f"{_self.data_path}raw/game_history.csv")  

    @st.cache_data(ttl=CACHE_TTL)
    def load_compile_stats(_self) -> pd.DataFrame:
        return pd.read_csv(f"{_self.data_path}targeted/compile_stats.csv") 

    @st.cache_data(ttl=CACHE_TTL)
    def load_goalkeepers(_self) -> pd.DataFrame:
        return pd.read_csv(f"{_self.data_path}targeted/goalkeepers_data.csv")
