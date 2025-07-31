import streamlit as st  # библиотека для создания веб-приложения
from ui.core.base import BaseView  # базовый класс представления
from typing import Any, Dict  # типизация стандартной библиотеки
from ui.models.data_loader import DataLoader
from app.src.preprocessing import *  # импорт всех необходимых функций валидации и обработки данных
from datetime import datetime

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
        if "lock_navigation" not in st.session_state:
            st.session_state.lock_navigation = False
        if "run_update" not in st.session_state:
            st.session_state.run_update = False

        if st.session_state.run_update:
            self._do_update()   # your data loading / validation logic
            # clear flags and force one more rerun so UI resets
            st.session_state.run_update = False
            st.session_state.lock_navigation = False
            st.rerun()

        # Выводим заголовок страницы, если он задан
        st.title(self.data.get("title", ""))
        # Выводим описание под заголовком
        st.write(self.data.get("description", ""))
        # Выводим инструкцию или подсказку по использованию
        st.write(self.data.get("instruction", ""))

        # Отображение времени последнего обновления
        last_update_file = 'data/raw/game_history.csv'
        if os.path.exists(last_update_file):
            last_update_time = datetime.fromtimestamp(os.path.getmtime(last_update_file)).strftime("%d-%m-%Y %H:%M")
            st.caption(f"🕒 Последнее обновление данных: {last_update_time}")

        def arm_update():
            st.session_state.lock_navigation = True
            st.session_state.run_update = True

        st.button(
            "🔄 Обновить и валидировать данные",
            on_click=arm_update,
            disabled=st.session_state.lock_navigation
        )

    def _do_update(self):
        # Плейсхолдеры для отображения этапов
        load_status = st.empty()
        validate_status = st.empty()
        progress_bar = st.progress(0)
        # ==== ЗАГРУЗКА ДАННЫХ ====
        with st.spinner("Загрузка и предобработка данных..."):
            total_steps = 7
            step = 1

            load_status.info(f"Загрузка данных: шаг {step}/{total_steps} — амплуа игроков")
            df_amplua = DataLoader.fetch_players()
            progress_bar.progress(step / total_steps)

            step += 1
            load_status.info(f"Загрузка данных: шаг {step}/{total_steps} — возраст игроков")
            df_player_age = pd.read_csv("data/raw/players_age.csv")
            progress_bar.progress(step / total_steps)

            step += 1
            load_status.info(f"Загрузка данных: шаг {step}/{total_steps} — статистика игроков")
            df_player_stats = DataLoader.fetch_player_stats()
            progress_bar.progress(step / total_steps)

            step += 1
            load_status.info(f"Загрузка данных: шаг {step}/{total_steps} — история игр")
            df_games_history = DataLoader.fetch_games_history()
            progress_bar.progress(step / total_steps)

            step += 1
            load_status.info(f"Загрузка данных: шаг {step}/{total_steps} — голы и передачи")
            df_goals_passes = DataLoader.fetch_game_goals_and_passes()
            progress_bar.progress(step / total_steps)

            step += 1
            load_status.info(f"Загрузка данных: шаг {step}/{total_steps} — события вратарей")
            df_keeper_events = DataLoader.fetch_keeper_events()
            progress_bar.progress(step / total_steps)

            step += 1
            load_status.info(f"Загрузка данных: шаг {step}/{total_steps} — данные +/-")
            df_plus_minus = DataLoader.fetch_game_plus_minus()
            progress_bar.progress(step / total_steps)

        load_status.success("✅ Все данные загружены")

        # ==== ВАЛИДАЦИЯ ====
        with st.spinner("Валидация и обработка данных..."):
            validation_steps = 16
            step = 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — подмена и форматирование")
            replace_data("data/raw")
            re_format_file("data/raw")
            replace_zero_with_nan("data/raw")
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — удаление некорректных игр")
            df_player_game = remove_invalid_games(df_player_stats)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — фильтрация игр")
            df_player_game = process_game_data(df_player_game)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — фильтрация игроков без амплуа")
            df_player_game = process_player_amplua(df_amplua, df_player_game)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — корректировка амплуа")
            compile_stats = check_and_modify_amplua(df_player_game)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — фильтрация событий вратарей")
            # goalk_data = remove_inconsistent_games(df_keeper_events)
            goalk_filt, compile_stats = filter_goalkeeper_event_by_compile_stats(df_keeper_events, compile_stats)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — объединение с событиями вратарей")
            compile_stats = merge_goalkeeper_events(compile_stats, goalk_filt)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — удаление пустых строк")
            compile_stats = remove_empty_rows(compile_stats)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — валидация +/-")
            pm_clean = validation_for_pm(df_plus_minus)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — фильтрация игр без +/-")
            df_pm_filtered = remove_PL_game(compile_stats, pm_clean)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — расчёт суммы +/-")
            pm_sums = calculate_plus_minus(pm_clean)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — добавление +/-")
            compile_stats = add_plus_minus(df_pm_filtered, pm_sums)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — удаление игр без времени")
            compile_stats = remove_games_with_missing_time(compile_stats)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — добавление голов и пасов")
            compile_stats = add_goals_and_assists(compile_stats, df_goals_passes)
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — сохранение промежуточных файлов")
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
            progress_bar.progress(step / validation_steps)

            step += 1
            validate_status.info(f"Валидация: шаг {step}/{validation_steps} — финальная очистка")
            clean_compile_stats('data/targeted/compile_stats.csv')
            create_goalkeepers_table(
                'data/targeted/compile_stats.csv',
                df_goals_passes,
                'data/targeted/goalkeepers_data.csv'
            )
            progress_bar.progress(step / validation_steps)

        validate_status.success("✅ Валидация завершена")
        st.success("Все данные успешно загружены и провалидированы.")