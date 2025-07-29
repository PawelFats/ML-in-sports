import streamlit as st  # библиотека для создания веб-приложения
from ui.core.base import BaseView  # базовый класс представления
from typing import Any, Dict  # типизация стандартной библиотеки
from ui.models.data_loader import DataLoader
from app.src.preprocessing import *  # импорт всех необходимых функций валидации и обработки данных

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

        if st.button("🔄 Обновить и валидировать данные"):
            # 1. Загрузка данных
            with st.spinner("Загрузка и предобработка данных..."):
                results = {}

                df_amplua = DataLoader.fetch_players()
                st.success("✅ Амплуа игроков загружены")

                age_file = "data/raw/players_age.csv"
                df_player_age = pd.read_csv(age_file)
                st.success(f"✅ Возраст игроков загружен")

                df_player_stats = DataLoader.fetch_player_stats()
                st.success("✅ Статистика игроков загружена")

                df_games_history = DataLoader.fetch_games_history()
                st.success("✅ История игр загружена")

                df_goals_passes = DataLoader.fetch_game_goals_and_passes()
                st.success("✅ Голы и пассы загружены")

                df_keeper_events = DataLoader.fetch_keeper_events()
                st.success("✅ События вратарей загружены")

                df_plus_minus = DataLoader.fetch_game_plus_minus()
                st.success("✅ События плюс/минус загружены")

            # 2. Валидация и обработка
            with st.spinner("Валидация данных..."):
                replace_data("data/raw")
                re_format_file("data/raw")
                replace_zero_with_nan("data/raw")
                # Шаг 1: удалить некорректные игры
                df_player_game = remove_invalid_games(df_player_stats)

                # Шаг 2: удалить игры с недостатком данных
                df_player_game = process_game_data(df_player_game)

                # Шаг 3: отфильтровать игроков без амплуа
                df_player_game = process_player_amplua(df_amplua, df_player_game)

                # Шаг 4: корректировка амплуа у игроков
                compile_stats = check_and_modify_amplua(df_player_game)

                # Шаг 5: фильтрация событий вратарей по последовательности игр
                goalk_data = remove_inconsistent_games(df_keeper_events)
                goalk_filt, compile_stats = filter_goalkeeper_event_by_compile_stats(goalk_data, compile_stats)

                # Шаг 6: объединение событий вратарей с общей статистикой
                compile_stats = merge_goalkeeper_events(compile_stats, goalk_filt)

                # Шаг 7: добавление возраста игроков
                # df_player_age должен быть подготовлен заранее или загружен отдельно
                #compile_stats = add_age_to_players_stats(compile_stats, df_player_age)

                # Шаг 8: удаление пустых строк (кроме вратарей)
                compile_stats = remove_empty_rows(compile_stats)

                # Шаг 9: очистка данных плюс/минус
                pm_clean = validation_for_pm(df_plus_minus)

                # Шаг 10: удалить игры без событий +/-
                df_pm_filtered = remove_PL_game(compile_stats, pm_clean)

                # Шаг 11: вычисление сумм +/-
                pm_sums = calculate_plus_minus(pm_clean)

                # Шаг 12: добавление значений +/-
                compile_stats = add_plus_minus(df_pm_filtered, pm_sums)

                # Шаг 13: удалить аномалии, игры без времени на льду
                compile_stats = remove_games_with_missing_time(compile_stats)

                # Шаг 14: добавить голы и передачи
                compile_stats = add_goals_and_assists(compile_stats, df_goals_passes)

                # Шаг 15: фильтрация событий вратарей и запись итоговых файлов
                filter_goalkeeper_events(
                    compile_stats,
                    df_keeper_events,
                    'data/interim/filtered_goalkeeper_event.csv',
                    'data/interim/goalkeeper_replace.csv'
                )
                del_swap_goalk(
                    'data/interim/filtered_goalkeeper_event.csv',
                    'data/interim/goalkeeper_replace.csv',
                    compile_stats,
                    'data/interim/filtered_goalkeeper_event_filtered.csv',
                    'data/targeted/compile_stats.csv'
                )

                # Финальная очистка целевого файла
                clean_compile_stats('data/targeted/compile_stats.csv')

                # Создание таблицы статистики вратарей
                create_goalkeepers_table(
                    'data/targeted/compile_stats.csv',
                    df_goals_passes,
                    'data/targeted/goalkeepers_data.csv'
                )

            st.write("Все данные успешно загружены и провалидированы.")
