import streamlit as st
from ui import ui_pages  

def main():
    st.set_page_config(page_title="Анализ хоккейной статистики", layout="wide")

    st.sidebar.title("Меню")

    # Статичное меню с radio-кнопками
    page = st.sidebar.radio("Выберите раздел:", 
                            ["Главная", "Статистика", "Рейтинги", "Графики", "Статистика игроков"])

    # Вызов соответствующей функции страницы
    if page == "Главная":
        ui_pages.home()
    elif page == "Статистика":
        ui_pages.stats()
    elif page == "Рейтинги":
        ui_pages.rankings()
    elif page == "Графики":
        ui_pages.charts()
    elif page == "Статистика игроков":
        ui_pages.player_rt()

if __name__ == "__main__":
    main()
