import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ui.core.base import BaseView
from typing import Any, Dict, Optional
from ui.models.player_division_analysis_model import PlayerDivisionAnalysisModel


class PlayerDivisionAnalysisView(BaseView):
    """Представление для страницы анализа рейтингов игроков по дивизионам."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.error_message: Optional[str] = None
    
    def update(self, data: Dict[str, Any]) -> None:
        """Обновление данных представления."""
        self.data = data
        self.error_message = data.get('error_message')
    
    def render(self) -> None:
        """Отрисовка страницы с анализом рейтингов игроков по дивизионам."""
        st.title("Анализ рейтингов игроков по дивизионам")
        st.write("Оценка соответствия игроков их дивизионам на основе рейтингов (советский метод)")
        
        if self.error_message:
            st.error(self.error_message)
            return
        
        # Функция для перевода названий дивизионов на русский
        def translate_division(div):
            div_str = str(div)
            if div_str.startswith('Division '):
                num = div_str.replace('Division ', '')
                return f'Дивизион {num}'
            elif div_str.isdigit() or (div_str.replace('.', '').isdigit() and '.' in div_str):
                num = int(float(div_str))
                return f'Дивизион {num}'
            return div_str
        
        # Получаем доступные сезоны
        model = PlayerDivisionAnalysisModel()
        available_seasons = model.get_available_seasons()
        
        if not available_seasons:
            st.error("Не удалось загрузить список сезонов из данных")
            return
        
        # Выбор сезонов (можно выбрать несколько)
        selected_seasons = st.multiselect(
            "Выберите сезоны для анализа",
            options=available_seasons,
            default=[available_seasons[-1]] if available_seasons else [],
            key="player_division_seasons",
            help="Выберите один или несколько сезонов для анализа. Можно выбрать несколько для сравнения динамики показателей."
        )
        
        # Очищаем результаты при изменении списка сезонов
        if 'player_analysis_seasons_prev' in st.session_state:
            prev_seasons = set(st.session_state.get('player_analysis_seasons_prev', []))
            curr_seasons = set(selected_seasons)
            if prev_seasons != curr_seasons:
                # Удаляем результаты для сезонов, которые больше не выбраны
                if 'player_analysis_results' in st.session_state:
                    for season in list(st.session_state['player_analysis_results'].keys()):
                        if season not in selected_seasons:
                            del st.session_state['player_analysis_results'][season]
        st.session_state['player_analysis_seasons_prev'] = selected_seasons.copy() if selected_seasons else []
        
        # Настройка порога отклонения
        st.subheader("Настройки анализа")
        deviation_threshold = st.slider(
            "Порог отклонения (стандартных отклонений)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Игроки, рейтинг которых отклоняется от среднего более чем на это значение стандартных отклонений, считаются несоответствующими дивизиону"
        )
        
        # Очищаем результаты при изменении порога
        if 'player_analysis_threshold' in st.session_state and st.session_state.get('player_analysis_threshold') != deviation_threshold:
            if 'player_analysis_results' in st.session_state:
                del st.session_state['player_analysis_results']
        
        # Инициализируем словарь для хранения результатов по сезонам
        if 'player_analysis_results' not in st.session_state:
            st.session_state['player_analysis_results'] = {}
        
        # Проверяем, какие сезоны нужно пересчитать
        seasons_to_calculate = []
        if selected_seasons:
            for season in selected_seasons:
                stored_threshold = st.session_state.get('player_analysis_threshold')
                # Пересчитываем, если сезон еще не загружен или изменился порог
                if (season not in st.session_state['player_analysis_results'] or 
                    stored_threshold != deviation_threshold):
                    seasons_to_calculate.append(season)
        
        # Кнопка анализа
        if st.button("Проанализировать", key="analyze_player_divisions") or seasons_to_calculate:
            if seasons_to_calculate:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, season in enumerate(seasons_to_calculate):
                    status_text.text(f"Расчет рейтингов для сезона {season}... ({idx+1}/{len(seasons_to_calculate)})")
                    with st.spinner(f"Расчет рейтингов для сезона {season}..."):
                        result = model.get_player_ratings_by_season_division(season)
                    
                    if result.get('error'):
                        st.error(f"Ошибка для сезона {season}: {result['error']}")
                    else:
                        # Сохраняем результаты для сезона
                        st.session_state['player_analysis_results'][season] = {
                            'result': result,
                            'original_stats': result.get('division_stats', pd.DataFrame()).copy()
                        }
                    
                    progress_bar.progress((idx + 1) / len(seasons_to_calculate))
                
                st.session_state['player_analysis_threshold'] = deviation_threshold
                status_text.empty()
                progress_bar.empty()
                if seasons_to_calculate:
                    st.success(f"Анализ завершен для {len(seasons_to_calculate)} сезон(ов)!")
        
        # Показываем результаты, если есть выбранные сезоны
        if selected_seasons and 'player_analysis_results' in st.session_state:
            # Слайдер для фиктивной корректировки результатов (применяется ко всем сезонам)
            st.subheader("🎛️ Корректировка результатов (для демонстрации)")
            adjustment = st.slider(
                "Корректировка результатов (%)",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                help="Фиктивная корректировка: при увеличении уменьшается количество игроков, несоответствующих дивизиону. Положительные значения = улучшение, отрицательные = ухудшение. Влияет только на отображаемые данные. Применяется ко всем выбранным сезонам."
            )
            
            # Вычисляем скорректированный порог отклонения
            adjusted_deviation_threshold = deviation_threshold * (1 + adjustment / 100)
            
            # Собираем данные для графиков динамики
            if len(selected_seasons) > 1:
                st.subheader("📈 Динамика показателей по сезонам")
                
                # Подготавливаем данные для графиков
                dynamics_data = []
                for season in sorted(selected_seasons):
                    if season in st.session_state['player_analysis_results']:
                        result_data = st.session_state['player_analysis_results'][season]
                        original_stats = result_data['original_stats'].copy()
                        
                        # Применяем корректировку
                        if not original_stats.empty and adjustment != 0:
                            adjustment_factor = 1 - (adjustment / 100)
                            original_stats['outliers_total'] = (original_stats['outliers_total'] * adjustment_factor).round().astype(int)
                            original_stats['outliers_total'] = original_stats['outliers_total'].clip(lower=0)
                        
                        # Суммируем показатели по всем дивизионам
                        total_players = original_stats['num_players'].sum() if 'num_players' in original_stats.columns else 0
                        total_outliers = original_stats['outliers_total'].sum()
                        avg_outlier_pct = (total_outliers / total_players * 100) if total_players > 0 else 0
                        avg_rating = original_stats['avg_rating'].mean() if 'avg_rating' in original_stats.columns else 0
                        
                        dynamics_data.append({
                            'Сезон': season,
                            'Всего игроков': total_players,
                            'Игроков несоответствующих': total_outliers,
                            'Процент несоответствующих (%)': avg_outlier_pct,
                            'Средний рейтинг': avg_rating
                        })
                
                if dynamics_data:
                    dynamics_df = pd.DataFrame(dynamics_data)
                    
                    # График 1: Процент несоответствующих игроков по сезонам
                    fig_outliers_pct = px.line(
                        dynamics_df,
                        x='Сезон',
                        y='Процент несоответствующих (%)',
                        title='Динамика процента игроков, несоответствующих дивизионам',
                        markers=True,
                        labels={'Процент несоответствующих (%)': 'Процент (%)', 'Сезон': 'Сезон'}
                    )
                    fig_outliers_pct.update_traces(line=dict(width=3), marker=dict(size=10))
                    st.plotly_chart(fig_outliers_pct, use_container_width=True)
                    
                    # График 2: Количество несоответствующих игроков по сезонам
                    fig_outliers_count = px.line(
                        dynamics_df,
                        x='Сезон',
                        y='Игроков несоответствующих',
                        title='Динамика количества игроков, несоответствующих дивизионам',
                        markers=True,
                        labels={'Игроков несоответствующих': 'Количество игроков', 'Сезон': 'Сезон'}
                    )
                    fig_outliers_count.update_traces(line=dict(width=3), marker=dict(size=10))
                    st.plotly_chart(fig_outliers_count, use_container_width=True)
                    
                    # График 3: Средний рейтинг по сезонам
                    fig_avg_rating = px.line(
                        dynamics_df,
                        x='Сезон',
                        y='Средний рейтинг',
                        title='Динамика среднего рейтинга игроков по сезонам',
                        markers=True,
                        labels={'Средний рейтинг': 'Средний рейтинг', 'Сезон': 'Сезон'}
                    )
                    fig_avg_rating.update_traces(line=dict(width=3), marker=dict(size=10))
                    st.plotly_chart(fig_avg_rating, use_container_width=True)
                    
                    # Таблица с данными
                    st.subheader("Сводная таблица по сезонам")
                    st.dataframe(dynamics_df, use_container_width=True)
            
            # Показываем детальный анализ для каждого сезона
            for season in sorted(selected_seasons):
                if season in st.session_state['player_analysis_results']:
                    result_data = st.session_state['player_analysis_results'][season]
                    result = result_data['result']
                    
                    if not result.get('error'):
                        player_ratings = result.get('player_ratings', pd.DataFrame())
                        original_division_stats = result_data['original_stats'].copy()
                        
                        if original_division_stats.empty:
                            original_division_stats = result.get('division_stats', pd.DataFrame()).copy()
                        
                        # Применяем корректировку к статистике дивизионов
                        division_stats = original_division_stats.copy()
                        
                        if not division_stats.empty and adjustment != 0:
                            # Вычисляем коэффициент корректировки для количества outliers
                            adjustment_factor = 1 - (adjustment / 100)
                            
                            division_stats['outliers_low_count'] = (division_stats['outliers_low_count'] * adjustment_factor).round().astype(int)
                            division_stats['outliers_high_count'] = (division_stats['outliers_high_count'] * adjustment_factor).round().astype(int)
                            division_stats['outliers_total'] = division_stats['outliers_low_count'] + division_stats['outliers_high_count']
                            
                            # Ограничиваем значения, чтобы не было отрицательных
                            division_stats['outliers_low_count'] = division_stats['outliers_low_count'].clip(lower=0)
                            division_stats['outliers_high_count'] = division_stats['outliers_high_count'].clip(lower=0)
                            division_stats['outliers_total'] = division_stats['outliers_total'].clip(lower=0)
                        
                        if player_ratings.empty:
                            st.warning(f"Нет данных о рейтингах игроков для сезона {season}")
                        else:
                            # Информация о сезоне
                            st.subheader(f"📊 Анализ сезона {season}")
                            if adjustment != 0:
                                st.info(f"⚠️ **Применена корректировка результатов: {adjustment:+.0f}%**. Это фиктивное изменение для демонстрации, реальные данные не изменены.")
                            st.info(f"""
                            **Что мы анализируем:**
                            - Рейтинги игроков, рассчитанные советским методом (красный алгоритм)
                            - Средние рейтинги игроков в каждом дивизионе
                            - Игроки, которые сильно отклоняются от среднего рейтинга своего дивизиона
                            - Оценка соответствия игроков их текущим дивизионам
                            """)
                            
                            # Статистика по дивизионам
                            if not division_stats.empty:
                                st.subheader("Статистика по дивизионам")
                                
                                # Переводим названия дивизионов
                                division_stats_display = division_stats.copy()
                                division_stats_display['Дивизион'] = division_stats_display['division'].apply(translate_division)
                                
                                # Форматируем для отображения
                                display_cols = {
                                    'Дивизион': 'Дивизион',
                                    'avg_rating': 'Средний рейтинг',
                                    'std_rating': 'Стандартное отклонение',
                                    'median_rating': 'Медианный рейтинг',
                                    'min_rating': 'Минимальный рейтинг',
                                    'max_rating': 'Максимальный рейтинг',
                                    'num_players': 'Количество игроков',
                                    'outliers_total': 'Игроков несоответствующих дивизиону'
                                }
                                
                                display_df = division_stats_display[[col for col in display_cols.keys()]].copy()
                                display_df.columns = [display_cols[col] for col in display_df.columns]
                                
                                # Форматируем числовые значения
                                for col in ['Средний рейтинг', 'Стандартное отклонение', 'Медианный рейтинг', 
                                           'Минимальный рейтинг', 'Максимальный рейтинг']:
                                    if col in display_df.columns:
                                        display_df[col] = display_df[col].round(2)
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # График средних рейтингов по дивизионам
                                st.subheader("Средние рейтинги игроков по дивизионам")
                                fig_avg = px.bar(
                                    division_stats_display,
                                    x='Дивизион',
                                    y='avg_rating',
                                    title=f'Средний рейтинг игроков по дивизионам (сезон {season})',
                                    labels={'avg_rating': 'Средний рейтинг', 'Дивизион': 'Дивизион'},
                                    color='avg_rating',
                                    color_continuous_scale='Blues'
                                )
                                st.plotly_chart(fig_avg, use_container_width=True)
                                
                                # График количества несоответствующих игроков
                                st.subheader("Игроки, несоответствующие дивизиону")
                                fig_outliers = px.bar(
                                    division_stats_display,
                                    x='Дивизион',
                                    y='outliers_total',
                                    title=f'Количество игроков, сильно отклоняющихся от среднего рейтинга дивизиона (сезон {season})',
                                    labels={'outliers_total': 'Количество игроков', 'Дивизион': 'Дивизион'},
                                    color='outliers_total',
                                    color_continuous_scale='Reds'
                                )
                                st.plotly_chart(fig_outliers, use_container_width=True)
                            
                            # Детальный анализ по дивизионам
                            st.subheader("Детальный анализ по дивизионам")
                            
                            for _, div_stat in division_stats.iterrows():
                                div = int(div_stat['division'])
                                div_name = translate_division(div)
                                
                                with st.expander(f"{div_name} - Детальный анализ", expanded=False):
                                    div_players = player_ratings[player_ratings['division'] == div].copy()
                                    
                                    if div_players.empty:
                                        st.write("Нет данных об игроках в этом дивизионе")
                                        continue
                                    
                                    avg_rating = div_stat['avg_rating']
                                    std_rating = div_stat['std_rating']
                                    # Используем скорректированный порог отклонения
                                    threshold_low = avg_rating - adjusted_deviation_threshold * std_rating
                                    threshold_high = avg_rating + adjusted_deviation_threshold * std_rating
                            
                            # Игроки с низким рейтингом (ниже порога)
                            low_rated = div_players[div_players['player_rating'] < threshold_low].copy()
                            # Игроки с высоким рейтингом (выше порога)
                            high_rated = div_players[div_players['player_rating'] > threshold_high].copy()
                            
                            # Применяем корректировку к количеству outliers (если adjustment != 0)
                            if adjustment != 0:
                                adjustment_factor = 1 - (adjustment / 100)
                                # Ограничиваем количество показываемых outliers
                                max_low = max(0, int(len(low_rated) * adjustment_factor))
                                max_high = max(0, int(len(high_rated) * adjustment_factor))
                                
                                # Берем только первые N игроков (самых крайних)
                                if len(low_rated) > max_low:
                                    low_rated = low_rated.nsmallest(max_low, 'player_rating')
                                if len(high_rated) > max_high:
                                    high_rated = high_rated.nlargest(max_high, 'player_rating')
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Средний рейтинг", f"{avg_rating:.2f}")
                            with col2:
                                st.metric("Игроков ниже нормы", len(low_rated))
                            with col3:
                                st.metric("Игроков выше нормы", len(high_rated))
                            
                            # График распределения рейтингов
                            fig_dist = go.Figure()
                            
                            # Гистограмма всех игроков
                            fig_dist.add_trace(go.Histogram(
                                x=div_players['player_rating'],
                                name='Все игроки',
                                marker_color='lightblue',
                                opacity=0.7
                            ))
                            
                            # Вертикальные линии для среднего и порогов
                            fig_dist.add_vline(
                                x=avg_rating,
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"Среднее: {avg_rating:.2f}"
                            )
                            fig_dist.add_vline(
                                x=threshold_low,
                                line_dash="dot",
                                line_color="orange",
                                annotation_text=f"Нижний порог: {threshold_low:.2f}"
                            )
                            fig_dist.add_vline(
                                x=threshold_high,
                                line_dash="dot",
                                line_color="orange",
                                annotation_text=f"Верхний порог: {threshold_high:.2f}"
                            )
                            
                            fig_dist.update_layout(
                                title=f'Распределение рейтингов игроков в {div_name}',
                                xaxis_title='Рейтинг игрока',
                                yaxis_title='Количество игроков',
                                barmode='overlay'
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Таблица игроков с низким рейтингом
                            if not low_rated.empty:
                                st.write(f"**Игроки с рейтингом ниже нормы (ниже {threshold_low:.2f}):**")
                                low_rated_display = low_rated[['ID player', 'amplua', 'ID team', 'games', 'player_rating']].copy()
                                low_rated_display = low_rated_display.sort_values('player_rating')
                                low_rated_display['player_rating'] = low_rated_display['player_rating'].round(2)
                                low_rated_display.columns = ['ID игрока', 'Амплуа', 'ID команды', 'Игр', 'Рейтинг']
                                st.dataframe(low_rated_display, use_container_width=True)
                            
                            # Таблица игроков с высоким рейтингом
                            if not high_rated.empty:
                                st.write(f"**Игроки с рейтингом выше нормы (выше {threshold_high:.2f}):**")
                                high_rated_display = high_rated[['ID player', 'amplua', 'ID team', 'games', 'player_rating']].copy()
                                high_rated_display = high_rated_display.sort_values('player_rating', ascending=False)
                                high_rated_display['player_rating'] = high_rated_display['player_rating'].round(2)
                                high_rated_display.columns = ['ID игрока', 'Амплуа', 'ID команды', 'Игр', 'Рейтинг']
                                st.dataframe(high_rated_display, use_container_width=True)
                            
                            # Box plot для визуализации распределения
                            fig_box = px.box(
                                div_players,
                                y='player_rating',
                                title=f'Распределение рейтингов в {div_name}',
                                labels={'player_rating': 'Рейтинг игрока'}
                            )
                            fig_box.add_hline(y=avg_rating, line_dash="dash", line_color="green", 
                                            annotation_text=f"Среднее: {avg_rating:.2f}")
                            st.plotly_chart(fig_box, use_container_width=True)
                            
                            # Общая статистика
                            st.subheader("Общая статистика")
                            total_players = len(player_ratings)
                            total_outliers = division_stats['outliers_total'].sum() if not division_stats.empty else 0
                            outlier_percentage = (total_outliers / total_players * 100) if total_players > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Всего игроков", total_players)
                            with col2:
                                st.metric("Игроков несоответствующих дивизиону", total_outliers)
                            with col3:
                                st.metric("Процент несоответствующих", f"{outlier_percentage:.1f}%")
                            
                            if outlier_percentage > 20:
                                st.warning(f"⚠️ Высокий процент игроков ({outlier_percentage:.1f}%) не соответствует своему дивизиону. Возможно, требуется пересмотр распределения команд.")
                            elif outlier_percentage > 10:
                                st.info(f"ℹ️ Умеренный процент игроков ({outlier_percentage:.1f}%) не соответствует своему дивизиону.")
                            else:
                                st.success(f"✅ Низкий процент игроков ({outlier_percentage:.1f}%) не соответствует своему дивизиону. Распределение выглядит сбалансированным.")
        else:
            st.info("Выберите хотя бы один сезон для анализа.")

