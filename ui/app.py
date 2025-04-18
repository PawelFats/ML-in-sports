import streamlit as st
import os
import sys

# Определяем корневую папку проекта: берем родительскую директорию от текущей директории файла
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui import ui_pages

#python -m streamlit run app/ui/app.py

def main():
    st.set_page_config(page_title="Анализ хоккейной статистики", layout="wide")

    st.sidebar.title("Меню")

    # Статичное меню с radio-кнопками
    page = st.sidebar.radio("Выберите раздел:", 
                            ["Главная", 
                             "Таблица с данными за каждую игру", 
                             "Актуальные рейтинги команд", 
                             "ELO рейтинг команды за все время", 
                             "Рейтинг игроков (интегральный метод)", 
                             "Рейтинг игроков (совеский метод)",
                             "Байесовский метод"])

    # Вызов соответствующей функции страницы
    if page == "Главная":
        ui_pages.home()
    elif page == "Таблица с данными за каждую игру":
        ui_pages.stats()
    elif page == "Актуальные рейтинги команд":
        ui_pages.rankings()
    elif page == "ELO рейтинг команды за все время":
        ui_pages.charts()
    elif page == "Рейтинг игроков (интегральный метод)":
        ui_pages.player_rt_intg()
    elif page == "Рейтинг игроков (совеский метод)":
        ui_pages.player_rt_red()
    elif page == "Байесовский метод":
        ui_pages.bayesian_analysis()

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>© 2025 Project Author: Pavel Fatyanov, Novosibirsk State Technical University</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>Assistance provided by Maxim Bakaev and Elizaveta Ulederkina.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
