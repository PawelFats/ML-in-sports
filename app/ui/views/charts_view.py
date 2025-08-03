import streamlit as st  # библиотека Streamlit для веб-интерфейса
import plotly.express as px  # библиотека Plotly для построения графиков
from app.ui.core.base import BaseView  # базовый класс представления
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
        """Отрисовка страницы с выбором одной или нескольких команд и графиком их ELO рейтинга по времени."""
        st.title(self.title)

        if self.error_message:
            st.error(self.error_message)
            return

        # Множественный выбор команд
        selected_teams = st.multiselect(
            "Выберите одну или несколько команд:",
            self.teams,
            default=self.teams[:1],  # по умолчанию первая команда выбрана
            key="selected_teams"
        )

        if selected_teams:
            # Фильтруем данные по выбранным командам
            df_filtered = self.data[self.data['ID team'].isin(selected_teams)].copy()

            if not df_filtered.empty:
                df_filtered['date'] = pd.to_datetime(df_filtered['date'])

                # Строим график с разделением по команде (цвет линии)
                fig = px.line(
                    df_filtered,
                    x='date',
                    y='ELO',
                    color='ID team',  # столбец, по которому будут разные цвета и легенда
                    title="Рейтинги ELO выбранных команд по времени",
                    labels={'team_id': 'Команда', 'ELO': 'Рейтинг ELO', 'date': 'Дата'}
                )

                st.plotly_chart(fig)
            else:
                st.warning("Нет данных для выбранных команд.")
        else:
            st.info("Выберите хотя бы одну команду.")
