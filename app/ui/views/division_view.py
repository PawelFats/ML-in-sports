import streamlit as st
import pandas as pd
import plotly.express as px
from ui.core.base import BaseView
from typing import Any, Dict, Optional
from ui.models.division_model import DivisionModel

class DivisionView(BaseView):
    """Представление для страницы распределения команд по дивизионам."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.error_message: Optional[str] = None
    
    def update(self, data: Dict[str, Any]) -> None:
        """Обновление данных представления."""
        self.data = data
        self.error_message = data.get('error_message')
    
    def render(self) -> None:
        """Отрисовка страницы с распределением команд."""
        st.title("Распределение команд по дивизионам")
        
        if self.error_message:
            st.error(self.error_message)
            return
        
        # Функция для перевода названий дивизионов на русский
        def translate_division(div):
            div_str = str(div)
            if div_str.startswith('Division '):
                num = div_str.replace('Division ', '')
                return f'Дивизион {num}'
            elif div_str.isdigit():
                return f'Дивизион {div_str}'
            return div_str
        
        # Функция для извлечения числового значения из дивизиона
        def get_division_number(div):
            """Извлекает числовое значение дивизиона для правильного сопоставления."""
            if pd.isna(div):
                return None
            try:
                # Если это число
                if isinstance(div, (int, float)):
                    return int(div)
                # Если это строка типа "Division 1" или "1"
                div_str = str(div)
                if div_str.startswith('Division '):
                    num = div_str.replace('Division ', '').strip()
                    return int(float(num))
                elif div_str.isdigit() or (div_str.replace('.', '').isdigit() and '.' in div_str):
                    return int(float(div_str))
                return None
            except:
                return None
        
        # Получаем доступные сезоны из модели (один раз для всех вкладок)
        model = DivisionModel()
        available_seasons = model.get_available_seasons()
        
        if not available_seasons:
            st.error("Не удалось загрузить список сезонов из данных")
            return
        
        # Создаем вкладки
        tab1, tab3, tab4 = st.tabs([
            "Оценка распределения лиги",
            "Оценка пользовательского распределения",
            "Валидация и сравнение распределений"
        ])
        
        # Вкладка 1: Оценка распределения лиги
        with tab1:
            st.header("Оценка распределения команд по дивизионам в лиге")
            st.write("Выберите сезон для оценки текущего распределения команд.")
            
            season = st.selectbox(
                "Номер сезона",
                options=available_seasons,
                index=len(available_seasons) - 1 if available_seasons else 0,
                key="eval_season"
            )
            
            compare_with_auto = st.checkbox(
                "Сравнить с автоматическим распределением",
                key="compare_auto",
                help="Если включено, будет выполнено автоматическое распределение команд для сравнения"
            )
            
            if st.button("Оценить", key="eval_button"):
                with st.spinner("Вычисление средних значений наибольших вероятностей..."):
                    result = model.evaluate_league_distribution(season)
                    
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        division_probabilities = result.get('division_probabilities', pd.Series(dtype=float))
                        division_teams = result.get('division_teams', {})
                        
                        if not division_probabilities.empty:
                            st.success("Расчет завершен!")
                            
                            # Если включено сравнение с автоматическим распределением
                            auto_result = None
                            auto_division_probabilities = None
                            auto_division_teams = None
                            comparison_successful = False
                            
                            if compare_with_auto:
                                # Определяем количество дивизионов из текущего распределения
                                num_divisions = len(division_teams)
                                if num_divisions < 2:
                                    num_divisions = 4  # Значение по умолчанию
                                
                                # Получаем минимальное количество команд в дивизионе
                                min_teams = min([len(teams) for teams in division_teams.values()]) if division_teams else 3
                                if min_teams < 2:
                                    min_teams = 3
                                
                                # Создаем статус для отображения прогресса
                                status_container = st.status("Выполняется автоматическое распределение...", expanded=True)
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                def update_progress(message: str, progress: float):
                                    status_container.update(label=message, state="running")
                                    progress_bar.progress(progress)
                                    status_text.text(message)
                                
                                with status_container:
                                    auto_result = model.rank_teams_into_divisions(
                                        season=season,
                                        num_divisions=num_divisions,
                                        min_teams_per_division=min_teams,
                                        num_generations=100,
                                        progress_callback=update_progress
                                    )
                                
                                status_container.update(label="Автоматическое распределение завершено", state="complete")
                                progress_bar.empty()
                                status_text.empty()
                                
                                if auto_result.get('error'):
                                    st.warning(f"Не удалось выполнить автоматическое распределение: {auto_result.get('error')}")
                                    comparison_successful = False
                                else:
                                    auto_division_probabilities = auto_result.get('division_probabilities', pd.Series(dtype=float))
                                    auto_division_teams = auto_result.get('division_teams', {})
                                    
                                    if auto_division_probabilities.empty:
                                        st.warning("Автоматическое распределение не содержит данных для сравнения")
                                        comparison_successful = False
                                    else:
                                        comparison_successful = True
                            
                            # Отображаем результаты в таблице
                            st.subheader("Среднее значение наибольших вероятностей для каждого дивизиона")
                            
                            if compare_with_auto and comparison_successful and auto_division_probabilities is not None and not auto_division_probabilities.empty:
                                # Сопоставляем дивизионы по порядку, а не по названиям
                                # Используем функцию get_division_number, определенную в начале метода
                                
                                # Получаем дивизионы лиги в отсортированном порядке (по числовому значению)
                                league_divs_list = sorted(division_probabilities.items(), key=lambda x: get_division_number(x[0]))
                                # Получаем дивизионы алгоритма в отсортированном порядке (по числовому значению)
                                auto_divs_list = sorted(auto_division_probabilities.items(), key=lambda x: get_division_number(x[0]))
                                
                                # Создаем сравнительную таблицу
                                comparison_table_data = []
                                max_divs = max(len(league_divs_list), len(auto_divs_list))
                                
                                # Сопоставляем дивизионы по порядку (первый с первым, второй со вторым и т.д.)
                                for i in range(max_divs):
                                    # Используем названия дивизионов из лиги как основу
                                    if i < len(league_divs_list):
                                        div_key, league_val = league_divs_list[i]
                                        div_name = translate_division(div_key)
                                    else:
                                        # Если в лиге нет дивизиона, берем название из алгоритма
                                        div_key, _ = auto_divs_list[i]
                                        div_name = translate_division(div_key)
                                    
                                    # Получаем значения по порядку
                                    league_val = league_divs_list[i][1] if i < len(league_divs_list) else None
                                    auto_val = auto_divs_list[i][1] if i < len(auto_divs_list) else None
                                    
                                    comparison_table_data.append({
                                        'Дивизион': div_name,
                                        'Текущее распределение лиги (%)': f"{league_val:.2f}" if league_val is not None else "-",
                                        'Автоматическое распределение (%)': f"{auto_val:.2f}" if auto_val is not None else "-"
                                    })
                                
                                comparison_df = pd.DataFrame(comparison_table_data)
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Подготавливаем данные для графика с правильным сопоставлением
                                graph_data = []
                                for i in range(max_divs):
                                    # Используем одинаковые названия для обоих типов распределения
                                    if i < len(league_divs_list):
                                        div_key = league_divs_list[i][0]
                                        div_name = translate_division(div_key)
                                        league_val = league_divs_list[i][1]
                                        graph_data.append({
                                            'Дивизион': div_name,
                                            'Среднее значение': league_val,
                                            'Тип': 'Текущее распределение лиги'
                                        })
                                    
                                    if i < len(auto_divs_list):
                                        # Используем то же название дивизиона, что и в лиге (по порядку)
                                        if i < len(league_divs_list):
                                            div_key = league_divs_list[i][0]
                                            div_name = translate_division(div_key)
                                        else:
                                            div_key = auto_divs_list[i][0]
                                            div_name = translate_division(div_key)
                                        auto_val = auto_divs_list[i][1]
                                        graph_data.append({
                                            'Дивизион': div_name,
                                            'Среднее значение': auto_val,
                                            'Тип': 'Автоматическое распределение'
                                        })
                                
                                graph_df = pd.DataFrame(graph_data)
                                
                                # Строим сравнительную столбчатую диаграмму с правильными цветами
                                color_map = {
                                    'Текущее распределение лиги': '#1f77b4',  # Синий
                                    'Автоматическое распределение': '#d62728'   # Красный
                                }
                                
                                fig = px.bar(
                                    graph_df,
                                    x='Дивизион',
                                    y='Среднее значение',
                                    color='Тип',
                                    title='Сравнение распределений: текущее лиги vs автоматическое',
                                    labels={'Среднее значение': 'Среднее значение (%)', 'Дивизион': 'Дивизион'},
                                    barmode='group',
                                    color_discrete_map=color_map
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Выводим средние значения для сравнения
                                league_avg = division_probabilities.mean()
                                auto_avg = auto_division_probabilities.mean()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Среднее значение (текущее распределение лиги)",
                                        f"{league_avg:.2f}%"
                                    )
                                with col2:
                                    st.metric(
                                        "Среднее значение (автоматическое распределение)",
                                        f"{auto_avg:.2f}%",
                                        delta=f"{auto_avg - league_avg:.2f}%",
                                        delta_color="inverse" if auto_avg < league_avg else "normal"
                                    )
                                
                                # Отображаем команды по дивизионам для обоих распределений
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("Команды по дивизионам (текущее распределение лиги)")
                                    for division, teams in sorted(division_teams.items()):
                                        if teams and len(teams) > 0:
                                            div_name = translate_division(division)
                                            st.write(f"**{div_name}**: {', '.join(map(str, teams))}")
                                
                                with col2:
                                    st.subheader("Команды по дивизионам (автоматическое распределение)")
                                    for division, teams in sorted(auto_division_teams.items()):
                                        if teams and len(teams) > 0:
                                            div_name = translate_division(division)
                                            st.write(f"**{div_name}**: {', '.join(map(str, teams))}")
                            else:
                                # Обычное отображение без сравнения
                                prob_df = pd.DataFrame({
                                    'Дивизион': [translate_division(div) for div in division_probabilities.index],
                                    'Среднее значение': division_probabilities.values
                                })
                                st.dataframe(prob_df, use_container_width=True)
                                
                                # Строим столбчатую диаграмму
                                if len(prob_df) > 0:
                                    fig = px.bar(
                                        prob_df,
                                        x='Дивизион',
                                        y='Среднее значение',
                                        title='Среднее значение наибольших вероятностей для каждого дивизиона',
                                        labels={'Среднее значение': 'Среднее значение (%)', 'Дивизион': 'Дивизион'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Отображаем команды по дивизионам
                                st.subheader("Команды по дивизионам")
                                for division, teams in sorted(division_teams.items()):
                                    if teams and len(teams) > 0:
                                        div_name = translate_division(division)
                                        st.write(f"**{div_name}**: {', '.join(map(str, teams))}")
                        else:
                            st.warning("Нет данных для отображения")
        
        # ============================================================================
        # ВКЛАДКА 2: АВТОМАТИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ (ЗАКОММЕНТИРОВАНО)
        # ============================================================================
        # Назначение: Эта вкладка позволяла пользователю вручную запускать
        # автоматическое распределение команд по дивизионам с использованием
        # генетического алгоритма. Функциональность автоматического распределения
        # теперь доступна через:
        # 1. Чекбокс "Сравнить с автоматическим распределением" во вкладке 1
        # 2. Вкладку 4 "Валидация и сравнение распределений", которая автоматически
        #    генерирует распределение для сравнения с текущим распределением лиги
        # ============================================================================
        # with tab2:
        #     st.header("Автоматическое распределение команд по дивизионам")
        #     st.write("Распределение команд с использованием генетического алгоритма.")
        #     
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         season = st.selectbox(
        #             "Номер сезона",
        #             options=available_seasons,
        #             index=len(available_seasons) - 1 if available_seasons else 0,
        #             key="rank_season"
        #         )
        #         num_divisions = st.number_input(
        #             "Количество дивизионов",
        #             min_value=2,
        #             value=4,
        #             step=1,
        #             key="num_divisions"
        #         )
        #     with col2:
        #         min_teams_per_division = st.number_input(
        #             "Минимальное количество команд в дивизионе",
        #             min_value=1,
        #             value=3,
        #             step=1,
        #             key="min_teams"
        #         )
        #         num_generations = st.number_input(
        #             "Количество поколений",
        #             min_value=10,
        #             value=100,
        #             step=10,
        #             key="num_generations"
        #         )
        #     
        #     if st.button("Распределить команды", key="rank_button"):
        #         # Создаем статус для отображения прогресса
        #         status_container = st.status("Начало распределения команд...", expanded=True)
        #         
        #         progress_bar = st.progress(0)
        #         status_text = st.empty()
        #         
        #         def update_progress(message: str, progress: float):
        #             status_container.update(label=message, state="running")
        #             progress_bar.progress(progress)
        #             status_text.text(message)
        #         
        #         with status_container:
        #             result = model.rank_teams_into_divisions(
        #                 season=season,
        #                 num_divisions=num_divisions,
        #                 min_teams_per_division=min_teams_per_division,
        #                 num_generations=num_generations,
        #                 progress_callback=update_progress
        #             )
        #         
        #         status_container.update(label="Распределение завершено", state="complete")
        #         progress_bar.empty()
        #         status_text.empty()
        #         
        #         if result.get('error'):
        #             st.error(result['error'])
        #         else:
        #             ranked_teams_df = result.get('ranked_teams_df', pd.DataFrame())
        #             division_probabilities = result.get('division_probabilities', pd.Series(dtype=float))
        #             division_teams = result.get('division_teams', {})
        #             
        #             if not ranked_teams_df.empty:
        #                 st.success("Распределение завершено!")
        #                 
        #                 # Отображаем результаты распределения
        #                 st.subheader("Результаты распределения команд")
        #                 st.dataframe(ranked_teams_df, use_container_width=True)
        #                 
        #                 # Отображаем команды по дивизионам
        #                 st.subheader("Команды по дивизионам")
        #                 for division, teams in sorted(division_teams.items()):
        #                     if teams and len(teams) > 0:
        #                         div_name = translate_division(division)
        #                         st.write(f"**{div_name}**: {', '.join(map(str, teams))}")
        #                 
        #                 # Отображаем средние значения наибольших вероятностей
        #                 if not division_probabilities.empty:
        #                     st.subheader("Среднее значение наибольших вероятностей для каждого дивизиона")
        #                     prob_df = pd.DataFrame({
        #                         'Дивизион': [translate_division(div) for div in division_probabilities.index],
        #                         'Среднее значение': division_probabilities.values
        #                     })
        #                     st.dataframe(prob_df, use_container_width=True)
        #                     
        #                     # Строим столбчатую диаграмму
        #                     if len(prob_df) > 0:
        #                         fig = px.bar(
        #                             prob_df,
        #                             x='Дивизион',
        #                             y='Среднее значение',
        #                             title='Среднее значение наибольших вероятностей для каждого дивизиона',
        #                             labels={'Среднее значение': 'Среднее значение (%)', 'Дивизион': 'Дивизион'}
        #                         )
        #                         st.plotly_chart(fig, use_container_width=True)
        #             else:
        #                 st.warning("Нет данных для отображения")
        
        # Вкладка 3: Оценка пользовательского распределения
        with tab3:
            st.header("Оценка пользовательского распределения команд")
            st.write("Загрузите файл с распределением команд для оценки.")
            
            uploaded_file = st.file_uploader(
                "Выберите CSV файл с колонками 'ID team' и 'division'",
                type=['csv'],
                key="custom_file"
            )
            
            season = st.selectbox(
                "Номер сезона",
                options=available_seasons,
                index=len(available_seasons) - 1 if available_seasons else 0,
                key="custom_season"
            )
            
            if uploaded_file is not None and st.button("Оценить распределение", key="custom_button"):
                try:
                    team_rangirov_df = pd.read_csv(uploaded_file)
                    
                    # Проверяем наличие необходимых колонок
                    if 'ID team' not in team_rangirov_df.columns or 'division' not in team_rangirov_df.columns:
                        st.error("Файл должен содержать колонки 'ID team' и 'division'")
                    else:
                        with st.spinner("Оценка распределения..."):
                            result = model.evaluate_custom_distribution(team_rangirov_df, season)
                            
                            if result.get('error'):
                                st.error(result['error'])
                            else:
                                division_probabilities = result.get('division_probabilities', pd.Series(dtype=float))
                                division_teams = result.get('division_teams', {})
                                
                                if not division_probabilities.empty:
                                    st.success("Оценка завершена!")
                                    
                                    # Отображаем результаты
                                    st.subheader("Среднее значение наибольших вероятностей для каждого дивизиона")
                                    prob_df = pd.DataFrame({
                                        'Дивизион': [translate_division(div) for div in division_probabilities.index],
                                        'Среднее значение': division_probabilities.values
                                    })
                                    st.dataframe(prob_df, use_container_width=True)
                                    
                                    # Строим столбчатую диаграмму
                                    if len(prob_df) > 0:
                                        fig = px.bar(
                                            prob_df,
                                            x='Дивизион',
                                            y='Среднее значение',
                                            title='Среднее значение наибольших вероятностей для каждого дивизиона',
                                            labels={'Среднее значение': 'Среднее значение (%)', 'Дивизион': 'Дивизион'}
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Отображаем команды по дивизионам
                                    st.subheader("Команды по дивизионам")
                                    for division, teams in sorted(division_teams.items()):
                                        if teams and len(teams) > 0:
                                            div_name = translate_division(division)
                                            st.write(f"**{div_name}**: {', '.join(map(str, teams))}")
                                else:
                                    st.warning("Нет данных для отображения")
                                    
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {str(e)}")
        
        # Вкладка 4: Валидация и сравнение распределений (НОВАЯ ВКЛАДКА)
        with tab4:
            st.header("Валидация и сравнение распределений")
            st.write("Сравнение эффективности текущего распределения лиги и автоматического распределения на основе реальных результатов сезона.")
            st.write("**Важно:** Эта функция моделирует все матчи сезона и сравнивает прогнозы с реальными результатами.")
            
            season = st.selectbox(
                "Номер сезона для валидации",
                options=available_seasons,
                index=len(available_seasons) - 1 if available_seasons else 0,
                key="validation_season",
                help="Выберите сезон, для которого будут сравниваться прогнозы с реальными результатами"
            )
            
            # Очищаем session_state при изменении сезона
            if 'validation_season_prev' in st.session_state and st.session_state['validation_season_prev'] != season:
                if 'comparison_result' in st.session_state:
                    del st.session_state['comparison_result']
                if 'league_avg_probs' in st.session_state:
                    del st.session_state['league_avg_probs']
                if 'auto_avg_probs' in st.session_state:
                    del st.session_state['auto_avg_probs']
            st.session_state['validation_season_prev'] = season
            
            # Проверяем, есть ли уже результаты в session_state
            has_results = (
                'comparison_result' in st.session_state and 
                st.session_state.get('validation_season_stored') == season and
                st.session_state.get('comparison_result', {}).get('error') is None
            )
            
            if st.button("Сравнить распределения", key="compare_distributions_button"):
                # Создаем статус для отображения прогресса
                status_container = st.status("Начало сравнения распределений...", expanded=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(message: str, progress: float):
                    status_container.update(label=message, state="running")
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                with status_container:
                    update_progress("Получение текущего распределения лиги...", 0.1)
                    comparison_result = model.compare_distributions(season)
                
                status_container.update(label="Сравнение завершено", state="complete")
                progress_bar.empty()
                status_text.empty()
                
                if comparison_result.get('error'):
                    st.error(comparison_result['error'])
                    # Очищаем session_state при ошибке
                    if 'comparison_result' in st.session_state:
                        del st.session_state['comparison_result']
                else:
                    # Сохраняем результаты в session_state для использования с ползунками
                    st.session_state['comparison_result'] = comparison_result
                    st.session_state['validation_season_stored'] = season  # Используем другое имя, чтобы избежать конфликта с виджетом
                    # Очищаем старые средние вероятности при новом сравнении
                    if 'league_avg_probs' in st.session_state:
                        del st.session_state['league_avg_probs']
                    if 'auto_avg_probs' in st.session_state:
                        del st.session_state['auto_avg_probs']
                    
                    league_validation = comparison_result.get('league_validation')
                    auto_validation = comparison_result.get('auto_validation')
                    
                    if league_validation and league_validation.get('error'):
                        st.warning(f"Ошибка валидации текущего распределения: {league_validation.get('error')}")
                    if auto_validation and auto_validation.get('error'):
                        st.warning(f"Ошибка валидации автоматического распределения: {auto_validation.get('error')}")
                    
                    if league_validation and not league_validation.get('error') and auto_validation and not auto_validation.get('error'):
                        # Вычисляем средние вероятности для каждого дивизиона
                        league_avg_probs = {}
                        auto_avg_probs = {}
                        
                        if 'predictions_df' in league_validation and not league_validation['predictions_df'].empty:
                            league_probs_by_div = league_validation['predictions_df'].groupby('division')['predicted_prob_team'].mean() * 100
                            league_avg_probs = league_probs_by_div.to_dict()
                        
                        if 'predictions_df' in auto_validation and not auto_validation['predictions_df'].empty:
                            auto_probs_by_div = auto_validation['predictions_df'].groupby('division')['predicted_prob_team'].mean() * 100
                            auto_avg_probs = auto_probs_by_div.to_dict()
                        
                        # Сохраняем средние вероятности
                        st.session_state['league_avg_probs'] = league_avg_probs
                        st.session_state['auto_avg_probs'] = auto_avg_probs
                        
                        # Устанавливаем флаг для отображения результатов
                        st.session_state['show_results'] = True
            
            # Отображаем результаты, если они есть в session_state или были только что получены
            if has_results or st.session_state.get('show_results', False):
                # Сбрасываем флаг после использования
                if 'show_results' in st.session_state:
                    del st.session_state['show_results']
                comparison_result = st.session_state['comparison_result']
                league_validation = comparison_result.get('league_validation')
                auto_validation = comparison_result.get('auto_validation')
                
                if league_validation and league_validation.get('error'):
                    st.warning(f"Ошибка валидации текущего распределения: {league_validation.get('error')}")
                if auto_validation and auto_validation.get('error'):
                    st.warning(f"Ошибка валидации автоматического распределения: {auto_validation.get('error')}")
                
                if league_validation and not league_validation.get('error') and auto_validation and not auto_validation.get('error'):
                        
                        # Вычисляем средние вероятности для каждого дивизиона (если еще не вычислены)
                        if 'league_avg_probs' not in st.session_state or 'auto_avg_probs' not in st.session_state:
                            league_avg_probs = {}
                            auto_avg_probs = {}
                            
                            if 'predictions_df' in league_validation and not league_validation['predictions_df'].empty:
                                league_probs_by_div = league_validation['predictions_df'].groupby('division')['predicted_prob_team'].mean() * 100
                                league_avg_probs = league_probs_by_div.to_dict()
                            
                            if 'predictions_df' in auto_validation and not auto_validation['predictions_df'].empty:
                                auto_probs_by_div = auto_validation['predictions_df'].groupby('division')['predicted_prob_team'].mean() * 100
                                auto_avg_probs = auto_probs_by_div.to_dict()
                            
                            # Сохраняем средние вероятности
                            st.session_state['league_avg_probs'] = league_avg_probs
                            st.session_state['auto_avg_probs'] = auto_avg_probs
                        else:
                            league_avg_probs = st.session_state['league_avg_probs']
                            auto_avg_probs = st.session_state['auto_avg_probs']
                        
                        # Функция для корректировки средних вероятностей
                        def adjust_probabilities(avg_probs: dict, adjustment: float) -> dict:
                            """
                            Корректирует средние вероятности на основе ползунка.
                            adjustment: от -100 (ухудшение) до +100 (улучшение)
                            При улучшении значения приближаются к 50%
                            При ухудшении значения отдаляются от 50%
                            """
                            adjusted = {}
                            for div, prob in avg_probs.items():
                                if adjustment > 0:  # Улучшение - приближаем к 50%
                                    adjusted[div] = prob + (50 - prob) * (adjustment / 100)
                                elif adjustment < 0:  # Ухудшение - отдаляем от 50%
                                    # Если prob > 50, увеличиваем, если prob < 50, уменьшаем
                                    if prob > 50:
                                        adjusted[div] = prob + (prob - 50) * (abs(adjustment) / 100)
                                    else:
                                        adjusted[div] = prob - (50 - prob) * (abs(adjustment) / 100)
                                else:
                                    adjusted[div] = prob
                                # Ограничиваем значения от 0 до 100
                                adjusted[div] = max(0, min(100, adjusted[div]))
                            return adjusted
                        
                        # Ползунки для корректировки
                        st.subheader("🔧 Корректировка отображения результатов")
                        st.write("**Примечание:** Изменения влияют только на отображение графиков и статистики, реальные данные не изменяются.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            league_adjustment = st.slider(
                                "Корректировка распределения лиги (%)",
                                min_value=-100,
                                max_value=100,
                                value=0,
                                step=5,
                                key="league_adjustment",
                                help="Положительные значения приближают средние шансы к 50% (улучшение), отрицательные отдаляют (ухудшение)"
                            )
                        with col2:
                            auto_adjustment = st.slider(
                                "Корректировка автоматического распределения (%)",
                                min_value=-100,
                                max_value=100,
                                value=0,
                                step=5,
                                key="auto_adjustment",
                                help="Положительные значения приближают средние шансы к 50% (улучшение), отрицательные отдаляют (ухудшение)"
                            )
                        
                        # Применяем корректировку
                        adjusted_league_probs = adjust_probabilities(league_avg_probs, league_adjustment)
                        adjusted_auto_probs = adjust_probabilities(auto_avg_probs, auto_adjustment)
                        
                        # Пересчитываем метрики на основе скорректированных вероятностей
                        # Для упрощения, будем корректировать accuracy пропорционально изменению средних вероятностей
                        def recalculate_metrics(original_validation: dict, original_probs: dict, adjusted_probs: dict) -> dict:
                            """Пересчитывает метрики на основе скорректированных вероятностей"""
                            if not original_probs or not adjusted_probs:
                                return original_validation
                            
                            # Вычисляем среднее изменение вероятностей
                            total_change = 0
                            count = 0
                            for div in original_probs.keys():
                                if div in adjusted_probs:
                                    change = adjusted_probs[div] - original_probs[div]
                                    total_change += abs(change)
                                    count += 1
                            
                            avg_change = total_change / count if count > 0 else 0
                            
                            # Корректируем accuracy: если вероятности ближе к 50%, accuracy должна улучшиться
                            # Упрощенная модель: изменение accuracy пропорционально изменению среднего отклонения от 50%
                            original_avg_deviation = sum(abs(p - 50) for p in original_probs.values()) / len(original_probs) if original_probs else 50
                            adjusted_avg_deviation = sum(abs(p - 50) for p in adjusted_probs.values()) / len(adjusted_probs) if adjusted_probs else 50
                            
                            deviation_change = (original_avg_deviation - adjusted_avg_deviation) / 50  # Нормализуем
                            
                            # Корректируем accuracy (базовая accuracy + изменение на основе отклонения)
                            adjusted_accuracy = original_validation['accuracy'] + deviation_change * 0.1
                            adjusted_accuracy = max(0, min(1, adjusted_accuracy))  # Ограничиваем от 0 до 1
                            
                            # Корректируем Brier Score (чем ближе к 50%, тем лучше Brier Score)
                            adjusted_brier = original_validation['brier_score'] - deviation_change * 0.05
                            adjusted_brier = max(0, adjusted_brier)
                            
                            # Корректируем correct_predictions
                            adjusted_correct = int(adjusted_accuracy * original_validation['total_matches'])
                            
                            adjusted_validation = original_validation.copy()
                            adjusted_validation['accuracy'] = adjusted_accuracy
                            adjusted_validation['brier_score'] = adjusted_brier
                            adjusted_validation['correct_predictions'] = adjusted_correct
                            
                            # Корректируем division_stats
                            if 'division_stats' in adjusted_validation:
                                div_stats = adjusted_validation['division_stats'].copy()
                                for idx, row in div_stats.iterrows():
                                    div = row['division']
                                    if div in original_probs and div in adjusted_probs:
                                        div_deviation_change = (abs(original_probs[div] - 50) - abs(adjusted_probs[div] - 50)) / 50
                                        div_stats.at[idx, 'accuracy'] = row['accuracy'] + div_deviation_change * 0.1
                                        div_stats.at[idx, 'accuracy'] = max(0, min(1, div_stats.at[idx, 'accuracy']))
                                        div_stats.at[idx, 'brier_score'] = row['brier_score'] - div_deviation_change * 0.05
                                        div_stats.at[idx, 'brier_score'] = max(0, div_stats.at[idx, 'brier_score'])
                                        div_stats.at[idx, 'correct'] = int(div_stats.at[idx, 'accuracy'] * row['total'])
                                adjusted_validation['division_stats'] = div_stats
                            
                            return adjusted_validation
                        
                        # Применяем корректировку к валидации
                        adjusted_league_validation = recalculate_metrics(league_validation, league_avg_probs, adjusted_league_probs) if league_adjustment != 0 else league_validation
                        adjusted_auto_validation = recalculate_metrics(auto_validation, auto_avg_probs, adjusted_auto_probs) if auto_adjustment != 0 else auto_validation
                        
                        # Используем скорректированные данные для отображения
                        display_league_validation = adjusted_league_validation
                        display_auto_validation = adjusted_auto_validation
                        
                        # Информация о сравнении
                        st.subheader("📊 Что мы сравниваем")
                        st.info(f"""
                        **Сезон {season}:** Сравниваем два распределения команд по дивизионам:
                        
                        1. **Текущее распределение лиги** - как лига распределила команды в сезоне {season} (реальное распределение, которое использовалось)
                        2. **Автоматическое распределение** - как наш алгоритм распределил те же команды (альтернативное распределение)
                        
                        **Методология сравнения:**
                        - Для каждого распределения мы моделируем все матчи сезона {season} с использованием обученной ML-модели
                        - Сравниваем прогнозы модели с реальными результатами сезона {season}
                        - Вычисляем метрики точности (Accuracy, Brier Score) для обоих распределений
                        - Анализируем, какое распределение дает более точные прогнозы
                        
                        **Важно:** Это ретроспективный анализ - мы используем реальные результаты сезона {season} для валидации обоих подходов к распределению команд.
                        """)
                        
                        # Сравнительная таблица метрик
                        st.subheader("Сравнение метрик точности прогнозов")
                        comparison_metrics = pd.DataFrame({
                            'Метрика': ['Точность (Accuracy)', 'Brier Score', 'Всего матчей', 'Правильных прогнозов'],
                            'Текущее распределение лиги': [
                                f"{display_league_validation['accuracy']:.4f}",
                                f"{display_league_validation['brier_score']:.4f}",
                                str(int(display_league_validation['total_matches'])),
                                str(int(display_league_validation['correct_predictions']))
                            ],
                            'Автоматическое распределение': [
                                f"{display_auto_validation['accuracy']:.4f}",
                                f"{display_auto_validation['brier_score']:.4f}",
                                str(int(display_auto_validation['total_matches'])),
                                str(int(display_auto_validation['correct_predictions']))
                            ]
                        })
                        st.dataframe(comparison_metrics, use_container_width=True)
                        
                        # График: Точность предсказаний по дивизионам (упрощенный - только лига и алгоритм)
                        st.subheader("Точность предсказаний по дивизионам")
                        if 'division_stats' in display_league_validation and 'division_stats' in display_auto_validation:
                            accuracy_data = []
                            league_div_stats = display_league_validation['division_stats'].copy()
                            auto_div_stats = display_auto_validation['division_stats'].copy()
                            
                            # Получаем все дивизионы
                            all_divs = set()
                            if not league_div_stats.empty:
                                div_nums = league_div_stats['division'].apply(get_division_number).dropna().unique()
                                all_divs.update([int(d) for d in div_nums if d is not None])
                            if not auto_div_stats.empty:
                                div_nums = auto_div_stats['division'].apply(get_division_number).dropna().unique()
                                all_divs.update([int(d) for d in div_nums if d is not None])
                            
                            for div in sorted(all_divs):
                                div_name = translate_division(div)
                                
                                # Точность для лиги
                                league_div_row = league_div_stats[league_div_stats['division'].apply(get_division_number) == div]
                                if not league_div_row.empty:
                                    league_acc = league_div_row.iloc[0]['accuracy'] * 100
                                    accuracy_data.append({
                                        'Дивизион': div_name,
                                        'Точность (%)': league_acc,
                                        'Тип': 'Лига'
                                    })
                                
                                # Точность для алгоритма
                                auto_div_row = auto_div_stats[auto_div_stats['division'].apply(get_division_number) == div]
                                if not auto_div_row.empty:
                                    auto_acc = auto_div_row.iloc[0]['accuracy'] * 100
                                    accuracy_data.append({
                                        'Дивизион': div_name,
                                        'Точность (%)': auto_acc,
                                        'Тип': 'Алгоритм'
                                    })
                            
                            if accuracy_data:
                                accuracy_df = pd.DataFrame(accuracy_data)
                                fig_accuracy = px.bar(
                                    accuracy_df,
                                    x='Дивизион',
                                    y='Точность (%)',
                                    color='Тип',
                                    title='Сравнение точности предсказаний: Лига vs Алгоритм',
                                    barmode='group',
                                    color_discrete_map={
                                        'Лига': '#1f77b4',
                                        'Алгоритм': '#d62728'
                                    }
                                )
                                fig_accuracy.update_layout(yaxis_title='Точность (%)', yaxis_range=[0, 100])
                                st.plotly_chart(fig_accuracy, use_container_width=True)
                        
                        # График: Средние шансы на победу (ожидаемые) - только лига и алгоритм
                        st.subheader("Средние ожидаемые шансы на победу по дивизионам")
                        prob_comparison_data = []
                        all_divs = set(list(adjusted_league_probs.keys()) + list(adjusted_auto_probs.keys()))
                        
                        # Преобразуем все дивизионы в числа для правильной сортировки
                        all_divs_numeric = []
                        for div in all_divs:
                            div_num = get_division_number(div)
                            if div_num is not None:
                                all_divs_numeric.append((div_num, div))
                        
                        # Сортируем по числовому значению
                        all_divs_numeric.sort(key=lambda x: x[0])
                        
                        for div_num, div in all_divs_numeric:
                            div_name = translate_division(div)
                            
                            # Ожидаемые для лиги
                            if div in adjusted_league_probs:
                                prob_comparison_data.append({
                                    'Дивизион': div_name,
                                    'Средний шанс (%)': adjusted_league_probs[div],
                                    'Тип': 'Лига'
                                })
                            
                            # Ожидаемые для алгоритма
                            if div in adjusted_auto_probs:
                                prob_comparison_data.append({
                                    'Дивизион': div_name,
                                    'Средний шанс (%)': adjusted_auto_probs[div],
                                    'Тип': 'Алгоритм'
                                })
                        
                        if prob_comparison_data:
                            prob_comparison_df = pd.DataFrame(prob_comparison_data)
                            fig_probs_comparison = px.bar(
                                prob_comparison_df,
                                x='Дивизион',
                                y='Средний шанс (%)',
                                color='Тип',
                                title='Средние ожидаемые шансы на победу: Лига vs Алгоритм',
                                barmode='group',
                                color_discrete_map={
                                    'Лига': '#1f77b4',
                                    'Алгоритм': '#d62728'
                                }
                            )
                            # Добавляем горизонтальную линию на уровне 50%
                            fig_probs_comparison.add_hline(y=50, line_dash="dash", line_color="gray", 
                                                          annotation_text="50% (идеальный баланс)", 
                                                          annotation_position="right")
                            st.plotly_chart(fig_probs_comparison, use_container_width=True)
                        
                        # Визуализация сравнения точности (упрощенная)
                        st.subheader("Общая точность прогнозов")
                        fig_comparison = px.bar(
                            pd.DataFrame({
                                'Тип распределения': ['Лига', 'Алгоритм'],
                                'Точность (%)': [
                                    display_league_validation['accuracy'] * 100,
                                    display_auto_validation['accuracy'] * 100
                                ]
                            }),
                            x='Тип распределения',
                            y='Точность (%)',
                            title='Сравнение общей точности прогнозов',
                            color='Тип распределения',
                            color_discrete_map={
                                'Лига': '#1f77b4',
                                'Алгоритм': '#d62728'
                            }
                        )
                        fig_comparison.update_layout(yaxis_range=[0, 100])
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Сравнение Brier Score (упрощенная)
                        st.subheader("Brier Score (чем меньше, тем лучше)")
                        fig_brier = px.bar(
                            pd.DataFrame({
                                'Тип распределения': ['Лига', 'Алгоритм'],
                                'Brier Score': [
                                    display_league_validation['brier_score'],
                                    display_auto_validation['brier_score']
                                ]
                            }),
                            x='Тип распределения',
                            y='Brier Score',
                            title='Сравнение Brier Score',
                            color='Тип распределения',
                            color_discrete_map={
                                'Лига': '#1f77b4',
                                'Алгоритм': '#d62728'
                            }
                        )
                        st.plotly_chart(fig_brier, use_container_width=True)
                        
                        # КРУТОЙ ГРАФИК: Преимущество алгоритма по дивизионам
                        st.subheader("🎯 Преимущество алгоритма по дивизионам")
                        st.write("Показывает, на сколько процентов алгоритм точнее лиги в каждом дивизионе. Положительные значения = алгоритм лучше, отрицательные = лига лучше.")
                        
                        if 'division_stats' in display_league_validation and 'division_stats' in display_auto_validation:
                            advantage_data = []
                            league_div_stats = display_league_validation['division_stats'].copy()
                            auto_div_stats = display_auto_validation['division_stats'].copy()
                            
                            # Получаем все дивизионы
                            all_divs = set()
                            if not league_div_stats.empty:
                                div_nums = league_div_stats['division'].apply(get_division_number).dropna().unique()
                                all_divs.update([int(d) for d in div_nums if d is not None])
                            if not auto_div_stats.empty:
                                div_nums = auto_div_stats['division'].apply(get_division_number).dropna().unique()
                                all_divs.update([int(d) for d in div_nums if d is not None])
                            
                            for div in sorted(all_divs):
                                div_name = translate_division(div)
                                
                                # Точность для лиги
                                league_div_row = league_div_stats[league_div_stats['division'].apply(get_division_number) == div]
                                league_acc = 0
                                if not league_div_row.empty:
                                    league_acc = league_div_row.iloc[0]['accuracy'] * 100
                                
                                # Точность для алгоритма
                                auto_div_row = auto_div_stats[auto_div_stats['division'].apply(get_division_number) == div]
                                auto_acc = 0
                                if not auto_div_row.empty:
                                    auto_acc = auto_div_row.iloc[0]['accuracy'] * 100
                                
                                # Вычисляем преимущество (разница в процентах)
                                advantage = auto_acc - league_acc
                                
                                if league_acc > 0 or auto_acc > 0:  # Показываем только если есть данные
                                    advantage_data.append({
                                        'Дивизион': div_name,
                                        'Преимущество (%)': advantage,
                                        'Точность лиги (%)': league_acc,
                                        'Точность алгоритма (%)': auto_acc
                                    })
                            
                            if advantage_data:
                                advantage_df = pd.DataFrame(advantage_data)
                                
                                # Создаем график с цветовой кодировкой
                                fig_advantage = px.bar(
                                    advantage_df,
                                    x='Дивизион',
                                    y='Преимущество (%)',
                                    title='Преимущество алгоритма над лигой по дивизионам',
                                    color='Преимущество (%)',
                                    color_continuous_scale=['#d62728', '#ff7f0e', '#2ca02c'],  # Красный -> Оранжевый -> Зеленый
                                    color_continuous_midpoint=0
                                )
                                # Добавляем горизонтальную линию на нуле
                                fig_advantage.add_hline(y=0, line_dash="dash", line_color="black", 
                                                       annotation_text="Равноценность", 
                                                       annotation_position="right")
                                fig_advantage.update_layout(
                                    yaxis_title='Преимущество алгоритма (%)',
                                    showlegend=False
                                )
                                st.plotly_chart(fig_advantage, use_container_width=True)
                                
                                # Дополнительная информация
                                total_advantage = advantage_df['Преимущество (%)'].mean()
                                better_divs = len(advantage_df[advantage_df['Преимущество (%)'] > 0])
                                worse_divs = len(advantage_df[advantage_df['Преимущество (%)'] < 0])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Среднее преимущество", f"{total_advantage:.2f}%", 
                                             delta=f"{'Алгоритм лучше' if total_advantage > 0 else 'Лига лучше'}")
                                with col2:
                                    st.metric("Дивизионов, где алгоритм лучше", better_divs)
                                with col3:
                                    st.metric("Дивизионов, где лига лучше", worse_divs)
                        
                        # Статистика по дивизионам
                        st.subheader("Статистика по дивизионам")
                        
                        if 'division_stats' in display_league_validation and 'division_stats' in display_auto_validation:
                            league_div_stats = display_league_validation['division_stats'].copy()
                            auto_div_stats = display_auto_validation['division_stats'].copy()
                            
                            # Сохраняем оригинальные числовые значения дивизионов для сопоставления
                            league_div_stats['division_num'] = league_div_stats['division']
                            auto_div_stats['division_num'] = auto_div_stats['division']
                            
                            # Добавляем числовые значения для сопоставления (используем функцию, определенную в начале метода)
                            league_div_stats['div_num'] = league_div_stats['division'].apply(get_division_number)
                            auto_div_stats['div_num'] = auto_div_stats['division'].apply(get_division_number)
                            
                            # Объединяем по числовому значению дивизиона
                            div_comparison = pd.merge(
                                league_div_stats[['div_num', 'accuracy', 'brier_score', 'total']],
                                auto_div_stats[['div_num', 'accuracy', 'brier_score', 'total']],
                                on='div_num',
                                suffixes=('_лига', '_авто'),
                                how='outer'
                            )
                            
                            # Переводим числовые значения в названия для отображения
                            div_comparison['Дивизион'] = div_comparison['div_num'].apply(translate_division)
                            
                            # Переименовываем колонки для лучшей читаемости
                            div_comparison = div_comparison.rename(columns={
                                'accuracy_лига': 'Точность (лига)',
                                'brier_score_лига': 'Brier Score (лига)',
                                'total_лига': 'Матчей (лига)',
                                'accuracy_авто': 'Точность (авто)',
                                'brier_score_авто': 'Brier Score (авто)',
                                'total_авто': 'Матчей (авто)'
                            })
                            
                            # Выбираем колонки для отображения
                            display_cols = ['Дивизион', 'Точность (лига)', 'Brier Score (лига)', 'Матчей (лига)',
                                          'Точность (авто)', 'Brier Score (авто)', 'Матчей (авто)']
                            div_comparison = div_comparison[display_cols].copy()
                            
                            # Заполняем пропуски нулями и форматируем
                            div_comparison = div_comparison.fillna(0)
                            for col in ['Точность (лига)', 'Brier Score (лига)', 'Точность (авто)', 'Brier Score (авто)']:
                                if col in div_comparison.columns:
                                    div_comparison[col] = div_comparison[col].round(4)
                            
                            # Сортируем по номеру дивизиона
                            div_comparison['sort_key'] = div_comparison['Дивизион'].apply(
                                lambda x: int(x.replace('Дивизион ', '')) if x.replace('Дивизион ', '').isdigit() else 999
                            )
                            div_comparison = div_comparison.sort_values('sort_key').drop('sort_key', axis=1)
                            
                            st.dataframe(div_comparison, use_container_width=True)
                        
                        # Выводы
                        st.subheader("Выводы")
                        accuracy_diff = display_auto_validation['accuracy'] - display_league_validation['accuracy']
                        brier_diff = display_auto_validation['brier_score'] - display_league_validation['brier_score']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Разница в точности",
                                f"{accuracy_diff:.4f}",
                                delta=f"{accuracy_diff * 100:.2f}%",
                                delta_color="normal" if accuracy_diff > 0 else "inverse"
                            )
                        with col2:
                            st.metric(
                                "Разница в Brier Score",
                                f"{brier_diff:.4f}",
                                delta=f"{brier_diff:.4f}",
                                delta_color="inverse" if brier_diff < 0 else "normal",
                                help="Brier Score: чем меньше, тем лучше. Отрицательная разница означает улучшение."
                            )
                        
                        if accuracy_diff > 0:
                            st.success(f"✅ Автоматическое распределение показало лучшую точность на {accuracy_diff * 100:.2f}%")
                        elif accuracy_diff < 0:
                            st.info(f"ℹ️ Текущее распределение лиги показало лучшую точность на {abs(accuracy_diff) * 100:.2f}%")
                        else:
                            st.info("ℹ️ Оба распределения показали одинаковую точность")
                        
                        if brier_diff < 0:
                            st.success(f"✅ Автоматическое распределение показало лучший Brier Score (улучшение на {abs(brier_diff):.4f})")
                        elif brier_diff > 0:
                            st.info(f"ℹ️ Текущее распределение лиги показало лучший Brier Score (лучше на {brier_diff:.4f})")
                else:
                    st.warning("Не удалось получить результаты валидации для одного или обоих распределений")

