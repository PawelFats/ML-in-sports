import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Функция для расчета средних значений для нападающих/защитников (amplua 9 и 10)
def calculate_overall_stats(mean_stats_pl):
    overall_stats_amplua_10 = mean_stats_pl[mean_stats_pl['amplua'] == 10].mean()
    overall_stats_amplua_9 = mean_stats_pl[mean_stats_pl['amplua'] == 9].mean()
    return overall_stats_amplua_10, overall_stats_amplua_9

# Функция для построения графика отклонений для игроков (нападающих/защитников)
def plot_player_deviation(players_df, amplua, player_ids):
    # Фильтрация игроков по выбранной амплуа и ID
    players = players_df[(players_df['amplua'] == amplua) & (players_df['ID player'].isin(player_ids))]
    
    # Расчет средних значений по всей выборке для заданных амплуа
    overall_stats_amplua_10, overall_stats_amplua_9 = calculate_overall_stats(players_df)
    overall_mean = overall_stats_amplua_10 if amplua == 10 else overall_stats_amplua_9

    # Вычисление отклонения (отбрасываем столбцы идентификаторов)
    deviation = players.drop(['ID player', 'amplua'], axis=1)
    deviation = (deviation - overall_mean) / np.abs(overall_mean) * 100

    # Сортировка отклонений по возрастанию для каждого игрока
    deviation_sorted = deviation.apply(lambda x: x.sort_values(), axis=1)

    # Построение графика
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f'Отклонение от среднего для игроков с amplua {amplua}')
    ax.set_xlabel('Показатель')
    ax.set_ylabel('Отклонение, %')
    ax.axhline(0, color='black', linewidth=2)

    # Строим графики для каждого столбца
    for col in deviation.columns:
        if col != 'ID player': # Исключаем столбцы ID team и ID player
            plt.plot(range(len(deviation_sorted)), deviation_sorted[col], marker='o', label=col)

    # Подписи по оси X - выводим ID игроков
    ax.set_xticks(range(len(deviation_sorted)), players['ID player'])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    
    return fig

# Функция для построения графика отклонений для вратарей (предполагается amplua 8)
def plot_goalk_deviation(players_df, player_ids):
    # Фильтрация игроков по выбранным ID (у вратарей можно добавить другой фильтр, если нужно)
    players = players_df[players_df['ID player'].isin(player_ids)]
    
    # Средние значения по всем игрокам (можно заменить на расчет для amplua 8, если в данных есть столбец)
    overall_stats = players_df.mean()

    # Вычисление отклонения
    deviation = players.drop(['ID player'], axis=1)
    deviation = ((deviation - overall_stats) / overall_stats) * 100

    # Если есть столбец 'MisG', умножаем его на -1
    if 'MisG' in deviation.columns:
        deviation['MisG'] *= -1

    # Построение графика в виде столбчатой диаграммы
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Отклонение от среднего для вратарей')
    ax.set_xlabel('Показатель')
    ax.set_ylabel('Отклонение, %')
    
    bar_width = 0.2
    index = np.arange(len(deviation))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    for i, col in enumerate(deviation.columns):
        if col != 'ID player':
            ax.bar(index + i * bar_width, deviation[col], bar_width,
                    color=colors[i % len(colors)], hatch=hatches[i % len(hatches)], label=col)
    
    ax.axhline(0, color='black', linewidth=2)
    ax.set_xticks(index + bar_width * (len(deviation.columns) / 2), players['ID player'])
    #ax.set_xticklabels(deviation.columns, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    return fig

# Загрузка данных (обратите внимание, что пути к файлам нужно скорректировать под вашу систему)
@st.cache_data
def load_data():
    mean_stats_pl = pd.read_csv(r'C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\processed\rating_last_time\mean_stats_pl.csv')
    mean_stats_goalk = pd.read_csv(r'C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\processed\rating_last_time\mean_stats_goalk.csv')
    return mean_stats_pl, mean_stats_goalk

def player_rt():
    st.title("Интерактивная визуализация статистики игроков")

    # Загрузка данных
    mean_stats_pl, mean_stats_goalk = load_data()
    
    # Выбор типа игрока
    player_type = st.radio("Выберите тип игрока для отображения статистики:",
                            ("Нападающие/Защитники", "Вратари"))

    if player_type == "Нападающие/Защитники":
        # Выбор амплуа
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
