import streamlit as st  # библиотека для создания веб-приложения
from ui.core.base import BaseView  # базовый класс представления
from typing import Any, Dict  # типизация стандартной библиотеки


class HomeView(BaseView):
    """Представление для главной страницы приложения."""
    
    def __init__(self):
        # Словарь с данными: заголовок, описание и инструкция
        self.data: Dict[str, str] = {}

    def update(self, data: Dict[str, str]) -> None:
        """
        Обновление данных представления.
        Ожидается словарь с ключами:
        - title: заголовок страницы
        - description: описание функциональности
        - instruction: инструкция по использованию
        """
        # Сохраняем переданные данные
        self.data = data

    def render(self) -> None:
        """
        Отрисовка главной страницы.
        Выводим заголовок, описание и инструкцию, если они есть.
        """
        # Выводим заголовок страницы, если он задан
        st.title(self.data.get("title", ""))
        # Выводим описание под заголовком
        st.write(self.data.get("description", ""))
        # Выводим инструкцию или подсказку по использованию
        st.write(self.data.get("instruction", ""))