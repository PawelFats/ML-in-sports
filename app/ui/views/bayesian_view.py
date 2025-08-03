import streamlit as st  # импортируем библиотеку для веб-приложений
from app.ui.core.base import BaseView  # базовый класс представления
from typing import Any, Dict, Optional  # типизация для удобства
import pandas as pd  # библиотека для работы с табличными данными


class BayesianView(BaseView):
    """Представление для страницы байесовского анализа."""

    def __init__(self):
        # Заголовок страницы
        self.title: str = ""
        # Данные для отображения (таблица pandas), изначально нет
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке, если что-то пошло не так
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """
        Обновление внутренних данных представления.
        Ожидается словарь с ключами "title", "data" и "error_message".
        """
        # Устанавливаем заголовок из переданных данных (или пустую строку по умолчанию)
        self.title = data.get("title", "")
        # Устанавливаем таблицу данных или None, если данных нет
        self.data = data.get("data")
        # Устанавливаем сообщение об ошибке или None
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """
        Отрисовка страницы в Streamlit.
        Выводим заголовок, затем либо сообщение об ошибке, либо таблицу данных.
        """
        # Выводим заголовок страницы
        st.title(self.title)

        # Если есть сообщение об ошибке, показываем его красным
        if self.error_message:
            st.error(self.error_message)
        # Если ошибок нет и есть данные, выводим их в виде таблицы
        elif self.data is not None:
            st.dataframe(self.data)
