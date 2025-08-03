import streamlit as st  # библиотека для создания веб-интерфейса
from app.ui.core.base import BaseView  # базовый класс для всех представлений
from typing import Any, Dict, Optional  # аннотации типов
import pandas as pd  # работа с табличными данными


class StatsView(BaseView):
    """Представление для страницы с детальной статистикой по играм."""
    
    def __init__(self):
        # Заголовок страницы (например, "Статистика игр")
        self.title: str = ""
        # Таблица с детальной статистикой по каждой игре (DataFrame)
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при загрузке данных
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """
        Обновление данных представления.
        Ожидает словарь с ключами:
        - title: заголовок страницы
        - data: pandas.DataFrame со статистикой по играм
        - error_message: сообщение об ошибке, если данные не получены
        """
        # Сохраняем заголовок страницы
        self.title = data.get("title", "")
        # Сохраняем таблицу статистики или None
        self.data = data.get("data")
        # Сохраняем сообщение об ошибке или None
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """
        Отрисовка страницы статистики.
        Выводим заголовок, затем либо сообщение об ошибке, либо таблицу данных.
        """
        # Выводим заголовок страницы
        st.title(self.title)

        # Если есть сообщение об ошибке, показываем его
        if self.error_message:
            st.error(self.error_message)
        # Если данные успешно загружены, показываем заголовок таблицы и саму таблицу
        elif self.data is not None:
            st.write("Таблица статистики игр:")
            st.dataframe(self.data)
