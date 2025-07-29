import streamlit as st  # библиотека Streamlit для веб-интерфейса
import plotly.express as px  # библиотека Plotly для построения графиков
from ui.core.base import BaseView  # базовый класс представления
from typing import Any, Dict, Optional, List  # типизация
import pandas as pd  # работа с табличными данными

class ChartsView(BaseView):
    """Представление для страницы графиков ELO рейтинга команды."""
    
    def __init__(self):
        # Заголовок страницы
        self.title: str = ""
        # Полная таблица с данными ELO для всех команд
        self.data: Optional[pd.DataFrame] = None
        # Список доступных команд (ID или имён)
        self.teams: List[int] = []
        # Данные только по выбранной команде
        self.selected_team_data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке, если загрузка не удалась
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """Обновление внутренних данных представления.

        Ожидаются ключи:
        - title: заголовок страницы
        - data: полный DataFrame с ELO по всем командам
        - teams: список идентификаторов/имён команд
        - selected_team_data: DataFrame с ELO по выбранной команде
        - error_message: сообщение об ошибке, если есть
        """
        self.title = data.get("title", "")
        self.data = data.get("data")
        self.teams = data.get("teams", [])
        self.selected_team_data = data.get("selected_team_data")
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """Отрисовка страницы с выбором команды и графиком её ELO рейтинга по времени."""
        # Выводим заголовок
        st.title(self.title)

        # Если есть ошибка, показываем её и выходим
        if self.error_message:
            st.error(self.error_message)
            return

        # Виджет для выбора команды из списка
        selected_team = st.selectbox(
            "Выберите команду:",
            self.teams,
            key="selected_team"
        )

        # Если данные для выбранной команды существуют и не пусты
        if self.selected_team_data is not None and not self.selected_team_data.empty:
            # создаём копию, чтобы не менять оригинал
            df = self.selected_team_data.copy()
            # преобразуем столбец 'date' в тип datetime для корректного построения графика
            df["date"] = pd.to_datetime(df["date"])
            # строим линейный график ELO по датам
            fig = px.line(
                df,
                x="date",
                y="ELO",
                title=f"Рейтинг ELO команды {selected_team} по времени"
            )
            # отображаем график в Streamlit
            st.plotly_chart(fig)
        # Если DataFrame существует, но пуст
        elif self.selected_team_data is not None:
            st.warning("Нет данных для выбранной команды.")
