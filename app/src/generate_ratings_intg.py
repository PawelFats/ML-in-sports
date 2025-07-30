import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

COLUMN_MAPPING = {
    'As': 'Assists',
    'BT': 'BlockedShots',
    'G': 'Goals',
    'Shot': 'Shots',
    'TB': 'ThrowsBy',
    'pm': 'P_M',
    'time': 'Time'
}

sns.set_style("whitegrid")  # единый стиль

# Функция для расчета средних значений для нападающих/защитников (amplua 9 и 10)
def calculate_overall_stats(mean_stats_pl):
    overall_stats_amplua_10 = mean_stats_pl[mean_stats_pl['amplua'] == 10].mean()
    overall_stats_amplua_9 = mean_stats_pl[mean_stats_pl['amplua'] == 9].mean()
    return overall_stats_amplua_10, overall_stats_amplua_9

def plot_player_deviation(players_df, amplua, player_ids,
                          bar_width=0.12,
                          figsize=(12, 6),
                          y_limits=(-200, 200)):
    """
    Унифицированная функция: стиль новой, логика и сигнатура старой.
    """
    # 1. Фильтрация нужных игроков по амплуа и ID
    df = players_df[(players_df['amplua'] == amplua) & (players_df['ID player'].isin(player_ids))].copy()
    if df.empty:
        raise ValueError("Нет подходящих игроков. Проверь 'amplua' и 'player_ids'.")

    # 2. Определяем метрики
    metrics = [col for col in df.columns if col not in ['ID player', 'amplua']]
    df_metrics = df[metrics].rename(columns=COLUMN_MAPPING)

    # 3. Средние значения по всем игрокам данной амплуа
    overall_10, overall_9 = calculate_overall_stats(players_df)
    mean_series = (overall_9 if amplua == 9 else overall_10).rename(COLUMN_MAPPING)[df_metrics.columns]

    # 4. Отклонение в процентах
    deviation = (df_metrics - mean_series) / mean_series.abs() * 100

    # 5. Настройка графика
    fig, ax = plt.subplots(figsize=figsize)
    n_players, n_metrics = deviation.shape
    x = np.arange(n_players)

    # Цвета и смещения
    palette = dict(zip(
        deviation.columns,
        sns.color_palette("bright", n_metrics)
    ))
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * bar_width

    # 6. Строим график: сортировка метрик внутри каждого игрока
    for i, pid in enumerate(df['ID player'].values):
        player_dev = deviation.iloc[i]
        sorted_metrics = player_dev.sort_values().index.tolist()

        for slot, metric in enumerate(sorted_metrics):
            ax.bar(
                x[i] + offsets[slot],
                player_dev[metric],
                width=bar_width,
                color=palette[metric],
                label=metric if i == 0 else "",
                alpha=1.0
            )

    # 7. Легенда (единожды)
    ax.legend(title='Metrics', title_fontsize=15,
              loc='upper left', bbox_to_anchor=(1, 1),
              frameon=False, fontsize=13)

    # 8. Оформление
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df['ID player'].astype(int), fontsize=12)
    ax.set_xlabel('ID игрока', fontsize=14)
    ax.set_ylabel('Отклонение, %', fontsize=14)

    if amplua == 10:
        ax.set_title(f'Отклонение от среднего для нападющих', fontsize=16)
    else:
        ax.set_title(f'Отклонение от среднего для защитников', fontsize=16)

    ax.set_ylim(y_limits)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    sns.despine(left=False, bottom=False)
    fig.tight_layout()

    return fig

def plot_goalk_deviation(players_df, player_ids):
    # Фильтрация игроков по выбранным ID
    players = players_df[players_df['ID player'].isin(player_ids)]
    
    # Средние значения по всем игрокам (без 'ID player')
    overall_stats = players_df.drop(columns=['ID player'], errors='ignore').mean()

    # Вычисление отклонения
    deviation = players.drop(columns=['ID player'], errors='ignore')
    deviation = ((deviation - overall_stats) / overall_stats) * 100

    # Если есть столбец 'MisG', инвертируем его
    if 'MisG' in deviation.columns:
        deviation['MisG'] *= -1

    # Построение графика
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Отклонение от среднего для вратарей', fontsize=16)
    ax.set_xlabel('ID игрока', fontsize=15)
    ax.set_ylabel('Отклонение, %', fontsize=15)
    
    bar_width = 0.2
    index = np.arange(len(deviation))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    drawn = set()

    for j in range(len(deviation)):
        # Сортируем метрики для игрока j по возрастанию отклонения
        sorted_cols = deviation.iloc[j].sort_values().index.tolist()

        for slot, col in enumerate(sorted_cols):
            ax.bar(
                index[j] + slot * bar_width,
                deviation.iloc[j][col],
                bar_width,
                color=colors[list(deviation.columns).index(col) % len(colors)],
                label=col if col not in drawn else ""
            )
            drawn.add(col)

    ax.axhline(0, color='black', linewidth=2)
    ax.set_xticks(index + bar_width * (len(deviation.columns) - 1) / 2)
    ax.set_xticklabels(players['ID player'].astype(int))

    if drawn:
        ax.legend(title='Метрики', title_fontsize=15, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    return fig

# Загрузка данных
@st.cache_data
def load_data():
    mean_stats_pl = pd.read_csv(r'data/processed/rating_last_time/mean_stats_pl.csv')
    mean_stats_goalk = pd.read_csv(r'data/processed/rating_last_time/mean_stats_goalk.csv')
    return mean_stats_pl, mean_stats_goalk

def player_rt_intg():
    st.title("Интерактивная визуализация статистики игроков")

    # Загрузка данных
    mean_stats_pl, mean_stats_goalk = load_data()
    
    # Выбор типа игрока
    player_type = st.radio("Выберите тип игрока для отображения статистики:",
                            ("Нападающие/Защитники", "Вратари"))

    if player_type == "Нападающие/Защитники":

        # Создаем словарь, сопоставляющий название позиции и числовое значение amplua
        player_position_options = {
            "Защитники": 9,
            "Нападающие": 10
        }
        # Выводим понятные названия для пользователя, а внутри получаем числовое значение
        selected_position = st.selectbox("Выберите амплуа:", options=list(player_position_options.keys()))
        amplua = player_position_options[selected_position]

        # Получение списка уникальных ID для выбранной амплуа
        available_ids = mean_stats_pl[mean_stats_pl['amplua'] == amplua]['ID player'].unique().tolist()
        # Мультиселект для выбора ID игроков
        player_ids = st.multiselect("Выберите ID игроков:", options=available_ids, default=available_ids[:4])
        
        if st.button("Построить график"):
            if player_ids:
                fig = plot_player_deviation(mean_stats_pl, amplua, player_ids)
                st.pyplot(fig)
            else:
                st.warning("Выберите хотя бы одного игрока для построения графика.")

    elif player_type == "Вратари":
        # Получаем список уникальных ID вратарей
        available_ids = mean_stats_goalk['ID player'].unique().tolist()
        player_ids = st.multiselect("Выберите ID вратарей:", options=available_ids, default=available_ids[:4])
        
        if st.button("Построить график"):
            if player_ids:
                fig = plot_goalk_deviation(mean_stats_goalk, player_ids)
                st.pyplot(fig)
            else:
                st.warning("Выберите хотя бы одного вратаря для построения графика.")
