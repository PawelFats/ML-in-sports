import streamlit as st
import pandas as pd

def rankings():
    st.title("Рейтинги команд")
    st.write("Здесь будет информация о рейтингах команд.")
    
    # Загрузка данных из CSV
    csv_path = r"C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\targeted\team_ratings_merge.csv"
    try:
        df = pd.read_csv(csv_path)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
