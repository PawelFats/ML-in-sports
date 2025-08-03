import streamlit as st  # библиотека для создания веб-интерфейса
from ui.core.base import BaseView  # базовый класс для всех представлений
from typing import Any, Dict, Optional  # стандартные аннотации типов
import pandas as pd  # работа с табличными данными


class PlayerRtIntgView(BaseView):
    """Представление для страницы рейтинга игроков (интегральный метод)."""

    def __init__(self):
        # Заголовок страницы
        self.title: str = ""
        # Таблица с рейтингами игроков, где каждый игрок — строка в DataFrame
        self.data: Optional[pd.DataFrame] = None
        # Сообщение об ошибке при загрузке или расчёте данных
        self.error_message: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        """
        Обновление данных представления.
        Ожидает словарь с ключами:
        - title: заголовок отображаемого раздела
        - data: pandas.DataFrame с рейтингами игроков
        - error_message: сообщение об ошибке, если расчёт не удался
        """
        # Извлекаем и сохраняем заголовок страницы
        self.title = data.get("title", "")
        # Сохраняем табличные данные рейтингов
        self.data = data.get("data")
        # Сохраняем возможное сообщение об ошибке
        self.error_message = data.get("error_message")

    def render(self) -> None:
        """
        Отрисовка интерфейса страницы.
        Показывает заголовок, затем либо ошибку, либо данные в виде таблицы.
        """
        # Выводим заголовок
        st.title(self.title)

        # Если есть ошибка, показываем её в виде красного сообщения
        if self.error_message:
            st.error(self.error_message)
        # Если данных нет, ничего не делаем, иначе выводим таблицу
        elif self.data is not None:
            st.dataframe(self.data)
