import streamlit as st
import pandas as pd

def stats():
    st.title("Статистика команд")
    
    # Здесь можно подключить данные из модуля проекта,
    # например, используя функцию из src/data/loader.py
    try:
        # Пример: загрузка данных из CSV-файла
        df = pd.read_csv(r"data/targeted/game_stats_one_r.csv")
        st.write("Таблица статистики", df)
    except Exception as e:
        st.error("Ошибка загрузки данных: " + str(e))
