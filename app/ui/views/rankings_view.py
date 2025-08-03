import streamlit as st  # библиотека для создания веб-интерфейса
from app.ui.core.base import BaseView  # базовый класс представления
from typing import Any, Dict, Optional  # стандартные типы для аннотаций
import pandas as pd  # работа с табличными данными через DataFrame


class RankingsView(BaseView):
    """Представление для страницы рейтинга команд."""
    
    def __init__(self):
        # Заголовок страницы (например, "Текущие рейтинги")
        self.title: str = ""
        # Описание или пояснения под заголовком
        self.description: str = ""
        # Табличные данные с рейтингом команд
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке, если данные не загрузились
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """
        Обновление данных представления.
        Ожидает словарь с ключами:
        - title: заголовок страницы
        - description: описание функционала
        - data: pandas.DataFrame с рейтингом команд
        - error_message: сообщение об ошибке
        """
        # Сохраняем заголовок и описание
        self.title = data.get("title", "")
        self.description = data.get("description", "")
        # Сохраняем таблицу рейтингов команд
        self.data = data.get("data")
        # Сохраняем возможный текст ошибки
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """
        Отрисовка страницы рейтингов.
        Показываем заголовок, описание, а затем данные или ошибку.
        """
        # Выводим заголовок страницы
        st.title(self.title)
        # Выводим описание или пояснение
        st.write(self.description)

        # Если есть сообщение об ошибке, показываем его
        if self.error_message:
            st.error(self.error_message)
        # Если данные есть, показываем их в виде таблицы
        elif self.data is not None:
            st.dataframe(self.data)
