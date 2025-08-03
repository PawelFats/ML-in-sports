import streamlit as st  # библиотека для создания веб-интерфейса
from ui.core.base import BaseView  # базовый класс для всех представлений
from typing import Any, Dict, Optional  # стандартные аннотации типов
import pandas as pd  # работа с табличными данными


class PlayerRtRedView(BaseView):
    """Представление для страницы рейтинга игроков ("советский" метод)."""

    def __init__(self):
        # Заголовок страницы
        self.title: str = ""
        # Таблица с рейтингами игроков по советской методике
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при загрузке или расчёте данных
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """
        Обновление данных представления.
        Ожидает словарь с ключами:
        - title: заголовок раздела
        - data: pandas.DataFrame с результатом расчётов
        - error_message: сообщение об ошибке, если расчёт не удался
        """
        # Сохраняем заголовок страницы
        self.title = data.get("title", "")
        # Сохраняем табличные данные рейтинга
        self.data = data.get("data")
        # Сохраняем текст ошибки или None
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """
        Отрисовка интерфейса страницы.
        Показывает заголовок, затем либо сообщение об ошибке, либо таблицу с данными.
        """
        # Выводим заголовок страницы
        st.title(self.title)

        # Если есть ошибка, выводим её красным
        if self.error_message:
            st.error(self.error_message)
        # Если данные существуют, выводим их в виде таблицы
        elif self.data is not None:
            st.dataframe(self.data)
