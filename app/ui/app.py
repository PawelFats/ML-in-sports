import streamlit as st
from ui import ui_pages  

#python -m streamlit run ui/app.py

def main():
    st.set_page_config(page_title="Анализ хоккейной статистики", layout="wide")

    st.sidebar.title("Меню")

    # Статичное меню с radio-кнопками
    page = st.sidebar.radio("Выберите раздел:", 
                            ["Главная", 
                             "Таблица с данными за каждую игру", 
                             "Акутальные рейтинги команд", 
                             "ELO рейтинг команды за все время", 
                             "Рейтинг игроков (интегральный метод)", 
                             "Рейтинг игроков (совеский метод)",
                             "Баесовский метод"])

    # Вызов соответствующей функции страницы
    if page == "Главная":
        ui_pages.home()
    elif page == "Таблица с данными за каждую игру":
        ui_pages.stats()
    elif page == "Акутальные рейтинги команд":
        ui_pages.rankings()
    elif page == "ELO рейтинг команды за все время":
        ui_pages.charts()
    elif page == "Рейтинг игроков (интегральный метод)":
        ui_pages.player_rt_intg()
    elif page == "Рейтинг игроков (совеский метод)":
        ui_pages.player_rt_red()
    elif page == "Баесовский метод":
        ui_pages.bayesian_analysis()

if __name__ == "__main__":
    main()
