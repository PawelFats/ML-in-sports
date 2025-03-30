import streamlit as st
import pandas as pd
import plotly.express as px

def charts():
    st.title("Графики статистики")
    
    # Пример загрузки данных
    try:
        df = pd.read_csv(r"C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\targeted\game_stats_one_r.csv")
        
        # Преобразуем столбец "date" в формат datetime, затем сортируем
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date", ascending=True)
        
    except Exception as e:
        st.error("Ошибка загрузки данных: " + str(e))
        return

    # Выбор команды через selectbox
    team = st.selectbox("Выберите команду:", df["ID team"].unique())
    team_data = df[df["ID team"] == team]

    # Построение интерактивного графика
    fig = px.line(team_data, x="date", y="ELO", title=f"Рейтинг ELO команды {team} по времени")
    st.plotly_chart(fig)
