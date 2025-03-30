import streamlit as st
from ui import pages

def main():
    st.set_page_config(page_title="Анализ хоккейной статистики", layout="wide")
    st.sidebar.title("Меню")

    # Выбор страницы через селектор
    page = st.sidebar.selectbox("Выберите раздел:", 
                                ["Главная", "Статистика", "Рейтинги", "Графики"])

    # Вызов соответствующей функции страницы
    if page == "Главная":
        pages.home()
    elif page == "Статистика":
        pages.stats()
    elif page == "Рейтинги":
        pages.rankings()
    elif page == "Графики":
        pages.charts()

if __name__ == "__main__":
    main()
