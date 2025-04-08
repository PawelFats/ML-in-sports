import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

METRICS = ['goals', 'assists', 'throws_by', 'shot_on_target', 'blocked_throws', 'p_m']
P_METRICS = ['p_goals', 'p_assists', 'p_throws_by', 'p_shot_on_target', 'p_blocked_throws', 'p_p_m']

@st.cache_data
def load_data():
    df_history = pd.read_csv(r"C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\raw\game_history.csv", sep=";")
    df_compile_stats = pd.read_csv(r'C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\targeted\compile_stats.csv')
    df_goalk_stats = pd.read_csv(r'C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\targeted\goalkeepers_data.csv')
    return df_history, df_compile_stats, df_goalk_stats

def rename_columns(df):
    """
    Переименовывает столбцы датафрейма в соответствии с заданной схемой.
    
    Если в датафрейме присутствует один из ключевых столбцов, он будет заменён на
    соответствующее название на русском языке.
    
    Аргументы:
        df (pandas.DataFrame): Исходный датафрейм с английскими названиями столбцов.
        
    Возвращает:
        pandas.DataFrame: Датафрейм с переименованными столбцами.
    """
    mapping = {
        'ID player': 'ID игрока',
        'amplua': 'амплуа',
        'games': 'игры',
        'goals': 'шайбы',
        'p_goals': 'о_ш',
        'assists': 'ассисты',
        'p_assists': 'о_а',
        'throws_by': 'броски_мимо',
        'p_throws_by': 'о_м',
        'shot_on_target': 'броски_в_створ',
        'p_shot_on_target': 'о_с',
        'blocked_throws': 'блок_броски',
        'p_blocked_throws': 'о_б',
        'p_m': 'п/м',
        'p_p_m': 'о_п/м',
        'player_rating': 'общий_рейтинг'
    }
    # Переименовываем столбцы, если они присутствуют в датафрейме
    df = df.rename(columns=mapping)
    return df


def calculate_player_stats(df, output_file=r"C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\processed\red_method\player_stats.csv"):
    """
    Считает суммарные достижения для каждого игрока за всё время.
    """
    df_filtered = df[df['amplua'].isin([9, 10])]
    
    player_stats = df_filtered.groupby(['ID player', 'amplua']).agg(
        games=('ID game', 'nunique'),
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        throws_by=('throws by', 'sum'),
        shot_on_target=('a shot on target', 'sum'),
        blocked_throws=('blocked throws', 'sum'),
        p_m=('p/m', 'sum'),
        #time=('total time on ice', 'sum')
    ).reset_index()

    player_stats.to_csv(output_file, index=False)
    
    return player_stats

def calculate_points(df, coefficient, amplua):
    """
    Считает очки игрока для каждого показателя с учетом его амплуа.
    """
    df_filtered = df[df['amplua'] == amplua].copy()
    
    for col in METRICS:
        df_filtered[f'p_{col}'] = ((df_filtered[col] + coefficient * df_filtered['games']) ** 2) / df_filtered['games']
    df_filtered['player_rating'] = df_filtered[P_METRICS].sum(axis=1)
    
    return round(df_filtered, 2)

def process_and_save(df, output_file=r"C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\processed\red_method\player_stats_with_points.csv"):
    """
    Рассчитывает и сохраняет актуальные рейтинги игроков.
    """
    df_stats = calculate_player_stats(df)
    
    # Вычисляем очки для каждого амплуа с соответствующими коэффициентами
    df_defenders = calculate_points(df_stats, 2/3, 9)
    df_forwards = calculate_points(df_stats, 1/6, 10)
    
    # Объединяем результаты
    df_final = pd.concat([df_defenders, df_forwards])
    
    # Если в исходном наборе имеются столбцы, которых нет в рассчитанном наборе,
    # можно скорректировать порядок столбцов (пример ниже)
    expected_cols = ['ID player', 'amplua', 'games', 'goals', 'p_goals', 
                     'assists', 'p_assists',  'throws_by', 'p_throws_by', 
                     'shot_on_target', 'p_shot_on_target', 'blocked_throws', 
                     'p_blocked_throws', 'p_m', 'p_p_m', 'player_rating']
    # Если какие-то столбцы отсутствуют — оставляем те, что есть
    cols = [c for c in expected_cols if c in df_final.columns]
    df_final = df_final[cols]

    df_final.to_csv(output_file, index=False)
    
    return df_final

def process_season(df_compile, df_history, season_id, player_ids=None, 
                   output_file=r"C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\processed\red_method\season_player_stats_with_points.csv"):
    """
    Функция:
      1. Фильтрует игры по сезону (по 'ID season') и, при необходимости, по выбранным игрокам.
      2. Выводит дополнительную информацию: число уникальных игр, игроков по амплуа, число команд.
      3. Вызывает функцию process_and_save для расчёта статистики и рейтингов.
    """
    player_ids = [int(pid) for pid in player_ids]
    # Фильтруем историю игр по нужному сезону
    df_history_season = df_history[df_history["ID season"] == int(season_id)]
    
    # Оставляем только игры, присутствующие в истории выбранного сезона
    df_season = pd.merge(df_compile, df_history_season[["ID"]], left_on="ID game", right_on="ID", how="inner")
    
    # Если указан список номеров игроков, оставляем только их
    if player_ids is not None and len(player_ids) > 0:
        df_season = df_season[df_season["ID player"].isin(player_ids)]
    
    # Вывод дополнительной информации
    unique_games = df_season['ID game'].nunique()
    unique_teams = df_season['ID team'].nunique()
    unique_players_amplua = df_season.groupby('amplua')['ID player'].nunique()
    
    #Сделать, чтобы был выбор конретных защитников и нападающих№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    
    st.write(f"Уникальных игр в сезоне: {unique_games}")
    st.write(f"Уникальных команд в сезоне: {unique_teams}")
    st.write("Уникальных игроков по амплуа:")
    for amplua, count in unique_players_amplua.items():
        if amplua == 8:
            st.write(f"  Вратари: {count}")
        elif amplua == 9:
            st.write(f"  Защитники: {count}")
        elif amplua == 10:
            st.write(f"  Атакующие: {count}")
    
    # Рассчитываем статистику и рейтинги
    df_final = process_and_save(df_season, output_file=output_file)
    
    return df_final

def plot_player_ratings(result_df, season_id):
    """
    Отрисовывает два графика:
      - Составной столбчатый график рейтингов по показателям.
      - График количества игр.
    """
    metric_labels = {
        'p_goals': 'голы',
        'p_assists': 'ассисты',
        'p_throws_by': 'мимо',
        'p_shot_on_target': 'в створ',
        'p_blocked_throws': 'блокированные',
        'p_p_m': 'п/м'
    }
    
    players = result_df['ID player'].astype(str)
    metrics = list(metric_labels.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    
    # 1. График: Составной столбчатый график (stacked bar chart) рейтингов в процентах
    # Рассчитываем процентное соотношение каждого показателя
    result_df_percent = result_df.copy()

    for metric in metrics:
        result_df_percent[metric] = (result_df_percent[metric] / result_df_percent['player_rating']) * 100

    fig1, ax1 = plt.subplots(figsize=(14,8))
    bottom = np.zeros(len(result_df_percent))
    for i, metric in enumerate(metrics):
        ax1.bar(players, result_df_percent[metric], bottom=bottom, color=colors[i], label=metric_labels[metric])
        bottom += result_df_percent[metric].values
    ax1.set_xlabel('ID игрока')
    ax1.set_ylabel('Рейтинговые очки')
    ax1.set_title(f'Структура рейтингов по показателям за сезон {season_id}')
    ax1.legend(title='Показатели', bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # График 2: количество игр
    fig2, ax2 = plt.subplots(figsize=(14,6))
    ax2.bar(players, result_df['games'], color='skyblue')
    ax2.set_xlabel('ID игрока')
    ax2.set_ylabel('Количество игр')
    ax2.set_title(f'Количество игр за сезон {season_id}')
    plt.xticks(rotation=45)

    # 3. Тепловая карта показателей
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    heatmap_data = result_df.set_index('ID player')[metrics].rename(columns=metric_labels)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={'label': 'Рейтинговые очки'},
        ax=ax3
    )
    ax3.set_title(f'Тепловая карта показателей за сезон {season_id}\n', fontsize=16, pad=20)
    ax3.set_xlabel('Показатели', fontsize=12)
    ax3.set_ylabel('ID игрока', fontsize=12)
    
    # 4. Группированный бар-чарт по метрикам
    fig4, ax4 = plt.subplots(figsize=(16, 8))
    melted_df = result_df.melt(
        id_vars=['ID player', 'games'], 
        value_vars=metrics,
        var_name='Metric',
        value_name='Value'
    )
    melted_df['Metric'] = melted_df['Metric'].map(metric_labels)
    
    sns.barplot(
        x='ID player',
        y='Value',
        hue='Metric',
        data=melted_df,
        palette=colors,
        edgecolor='w',
        ax=ax4
    )
    ax4.set_title(f'Распределение показателей по игрокам (сезон {season_id})', fontsize=16)
    ax4.set_xlabel('ID игрока', fontsize=12)
    ax4.set_ylabel('Рейтинговые очки', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Показатели', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Радарная диаграмма для топ-5 игроков
    #result_df['total_rating'] = result_df[metrics].sum(axis=1)
    top_players = result_df.nlargest(5, 'player_rating')
    labels = list(metric_labels.values())
    num_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    radar_colors = sns.color_palette("husl", 5)
    fig5, ax5 = plt.subplots(subplot_kw={'polar': True}, figsize=(10, 10))
    
    top_players = top_players.reset_index(drop=True)
    for i, (_, row) in enumerate(top_players.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]
        ax5.plot(angles, values, color=radar_colors[i], linewidth=2, 
                 label=f"ID {row['ID player']} (Σ={row['player_rating']:.1f})")
        ax5.fill(angles, values, color=radar_colors[i], alpha=0.1)
    
    ax5.set_theta_offset(np.pi / 2)
    ax5.set_theta_direction(-1)
    ax5.set_rlabel_position(30)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(labels, fontsize=10)
    ax5.tick_params(axis='y', labelsize=8)
    ax5.set_title('Топ-5 игроков по суммарному рейтингу\n', fontsize=16, pad=40)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10, frameon=True, shadow=True)
    ax5.spines['polar'].set_visible(False)
    ax5.grid(alpha=0.5, linestyle='--')
    
    # Возвращаем все созданные фигуры
    return fig1, fig2, fig3, fig4, fig5

def plot_team_ratings(df_compile, df_history, season_id=None, team_ids=None):
    """
    Функция:
      1. Загружает данные.
      2. При наличии season_id — фильтрует данные по сезону.
      3. При наличии team_ids — оставляет данные только для указанных команд.
      4. Группирует данные по игрокам, рассчитывает рейтинговые очки.
      5. Строит следующие графики:
           - Столбчатая диаграмма суммарного рейтинга команд.
           - Составной (stacked) бар-чарт с разбивкой по амплуа (защитники и атакующие).
           - Диаграмма рассеяния: зависимость суммарного рейтинга от числа уникальных игроков.
           - Составной (stacked) бар-чарт, где для каждой команды показано процентное соотношение вклада
             каждого показателя (голы, ассисты, мимо, в створ, блокированные, п/м) в общий рейтинг.
    """
    
    # Фильтрация данных по сезону
    if season_id is not None:
        df_history_season = df_history[df_history["ID season"] == int(season_id)]
        df = pd.merge(df_compile, df_history_season[["ID", "division"]], 
                      left_on="ID game", right_on="ID", how="inner")
    else:
        df = pd.merge(
            df_compile,
            df_history[["ID", "division"]],
            left_on="ID game",
            right_on="ID",
            how="left"
        )
    
    # Получаем информацию о дивизионе для команд
    df_team_div = df.groupby('ID team', as_index=False)['division'].first()
    
    # Фильтрация по выбранным командам, если заданы
    if team_ids is not None and len(team_ids) > 0:
        df = df[df["ID team"].isin(team_ids)]
    
    # Группировка данных по игрокам для расчёта статистики
    df_grouped = df.groupby(['ID player', 'amplua', 'ID team']).agg(
        games=('ID game', 'nunique'),
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        throws_by=('throws by', 'sum'),
        shot_on_target=('a shot on target', 'sum'),
        blocked_throws=('blocked throws', 'sum'),
        p_m=('p/m', 'sum')
    ).reset_index()
        
    # Разбиваем данные по амплуа: 9 – защитники, 10 – атакующие
    df_def = df_grouped[df_grouped['amplua'] == 9]
    df_for = df_grouped[df_grouped['amplua'] == 10]
    
    # Расчёт рейтингов для каждого амплуа
    df_def_calc = calculate_points(df_def, 2/3, 9)
    df_for_calc = calculate_points(df_for, 1/6, 10)
    
    # Объединяем данные игроков
    df_players = pd.concat([df_def_calc, df_for_calc])
    
    # Группировка по командам для получения суммарного рейтинга
    df_team = df_players.groupby('ID team').agg(
        team_rating=('player_rating', 'sum')
    ).reset_index()
    
    df_team['team_rating'] = round(df_team['team_rating'], 2)
    df_team = pd.merge(df_team, df_team_div, on='ID team', how='left')
    
    # 1. Столбчатая диаграмма суммарного рейтинга для команд
    fig_total, ax_total = plt.subplots(figsize=(12, 7))
    team_labels = df_team['ID team'].astype(str) + ' (Div ' + df_team['division'].astype(str) + ')'
    bars = ax_total.bar(team_labels, df_team['team_rating'], color='teal')
    if season_id is not None:
        ax_total.set_title(f'Суммарный рейтинг команд в сезоне {season_id}', fontsize=14)
    else:
        ax_total.set_title('Суммарный рейтинг команд за всё время', fontsize=14)
    ax_total.set_xlabel('ID команды', fontsize=12)
    ax_total.set_ylabel('Суммарный рейтинг', fontsize=12)
    plt.xticks(rotation=45)
    for bar, team_rating in zip(bars, df_team['team_rating']):
        height = bar.get_height()
        ax_total.text(bar.get_x() + bar.get_width() / 2, height, team_rating,
                      ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    
    # 2. Составной (stacked) бар-чарт с разбивкой по амплуа
    # Группируем по командам и амплуа, суммируя рейтинговые очки
    df_team_amplua = df_players.groupby(['ID team', 'amplua'])['player_rating'].sum().unstack(fill_value=0).reset_index()
    # Добавляем информацию о дивизионе
    df_team_amplua = pd.merge(df_team_amplua, df_team_div, on='ID team', how='left')
    team_labels_stacked = df_team_amplua['ID team'].astype(str) + ' (Div ' + df_team_amplua['division'].astype(str) + ')'
    
    fig_stacked, ax_stacked = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(df_team_amplua))
    # Определяем цвета и подписи для амплуа
    amplua_colors = {9: 'skyblue', 10: 'salmon'}
    amplua_labels = {9: 'Защитники', 10: 'Атакующие'}
    for amplua in [9, 10]:
        if amplua in df_team_amplua.columns:
            values = df_team_amplua[amplua]
            ax_stacked.bar(team_labels_stacked, values, bottom=bottom, 
                           color=amplua_colors[amplua], label=amplua_labels[amplua])
            bottom += values.values
    ax_stacked.set_title('Структура рейтинга по амплуа', fontsize=14)
    ax_stacked.set_xlabel('Команда', fontsize=12)
    ax_stacked.set_ylabel('Суммарный рейтинг', fontsize=12)
    ax_stacked.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 3. Диаграмма рассеяния: зависимость количества уникальных игроков от суммарного рейтинга команды
    # Считаем число уникальных игроков для каждой команды
    df_team_players = df_players.groupby('ID team')['ID player'].nunique().reset_index(name='num_players')
    df_scatter = pd.merge(df_team, df_team_players, on='ID team', how='left')
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 7))
    ax_scatter.scatter(df_scatter['team_rating'], df_scatter['num_players'], color='purple', s=100, alpha=0.7)
    for idx, row in df_scatter.iterrows():
        ax_scatter.text(row['team_rating'], row['num_players'], str(row['ID team']),
                        fontsize=9, ha='center', va='bottom')
    ax_scatter.set_xlabel("Суммарный рейтинг команды", fontsize=12)
    ax_scatter.set_ylabel("Количество уникальных игроков", fontsize=12)
    ax_scatter.set_title("Зависимость числа игроков от рейтинга команды", fontsize=14)
    
    plt.tight_layout()
    
    # 4. Составной бар-чарт с процентным соотношением вклада показателей в общий рейтинг команды
    # Определяем метрики и их подписи
    metric_labels = {
        'p_goals': 'голы',
        'p_assists': 'ассисты',
        'p_throws_by': 'мимо',
        'p_shot_on_target': 'в створ',
        'p_blocked_throws': 'блокированные',
        'p_p_m': 'п/м'
    }
    metrics = list(metric_labels.keys())
    
    # Группируем данные по командам для суммирования метрик
    df_team_metrics = df_players.groupby('ID team')[metrics].sum().reset_index()
    # Рассчитываем общую сумму метрик для каждой команды
    df_team_metrics['total'] = df_team_metrics[metrics].sum(axis=1)
    # Переводим каждую метрику в проценты от общего рейтинга
    for metric in metrics:
        df_team_metrics[metric] = df_team_metrics[metric] / df_team_metrics['total'] * 100
    
    # Добавляем информацию о дивизионе
    df_team_metrics = pd.merge(df_team_metrics, df_team_div, on='ID team', how='left')
    team_labels_metric = df_team_metrics['ID team'].astype(str) + ' (Div ' + df_team_metrics['division'].astype(str) + ')'
    
    fig_metric, ax_metric = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(df_team_metrics))
    colors_metric = {
        'p_goals': '#1f77b4',
        'p_assists': '#ff7f0e',
        'p_throws_by': '#2ca02c',
        'p_shot_on_target': '#d62728',
        'p_blocked_throws': '#9467bd',
        'p_p_m': '#8c564b'
    }
    
    for metric in metrics:
        values = df_team_metrics[metric]
        ax_metric.bar(team_labels_metric, values, bottom=bottom, 
                      color=colors_metric[metric], label=metric_labels[metric])
        bottom += values.values
    ax_metric.set_title("Процентное соотношение показателей в общем рейтинге команды", fontsize=14)
    ax_metric.set_xlabel("ID команды", fontsize=12)
    ax_metric.set_ylabel("Процентный вклад", fontsize=12)
    ax_metric.legend(title='Показатели', bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 7. Радарная диаграмма для топ-5 команд по суммарному рейтингу
    df_radar = df_players.groupby('ID team')[['p_goals','p_assists','p_throws_by','p_shot_on_target','p_blocked_throws','p_p_m','player_rating']].sum().reset_index()
    top_teams = df_radar.nlargest(5, 'player_rating')  # здесь можно выбрать любую метрику или суммарный рейтинг
    labels = ['голы', 'ассисты', 'мимо', 'в створ', 'блокированные', 'п/м']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # замыкаем круг

    fig_radar, ax_radar = plt.subplots(subplot_kw={'polar': True}, figsize=(10, 10))
    radar_colors = sns.color_palette("Set2", n_colors=top_teams.shape[0])
    top_teams = top_teams.reset_index(drop=True)

    for i, row in top_teams.iterrows():
        values = row[P_METRICS].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, color=radar_colors[i], linewidth=2, label=f"Команда ID {row['ID team']} ({round(row['player_rating'], 2)})")
        ax_radar.fill(angles, values, color=radar_colors[i], alpha=0.25)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax_radar.set_title("Радарная диаграмма для топ-5 команд", fontsize=14)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Возвращаем таблицу и все созданные графики
    return df_team, fig_total, fig_stacked, fig_scatter, fig_metric, fig_radar

def player_rt_red():
    st.title("Интерактивная визуализация статистики игроков")
    st.sidebar.header("Настройки")
    
    with st.spinner("Загрузка данных..."):
        df_history, df_compile_stats, df_goalk_stats = load_data()
    
    # Объединяем данные, чтобы брать только те сезоны, в которых есть игры из compile_stats
    df_merged = pd.merge(df_compile_stats, df_history, left_on="ID game", right_on="ID", how="inner")
    available_seasons = sorted(df_merged["ID season"].unique())
    
    action = st.sidebar.selectbox("Выберите действие", 
                                  ["Актуальный рейтинг игроков", "Статистика игроков за сезон", "Визуализация команд"])
    
    # 1. Общая статистика игроков
    if action == "Актуальный рейтинг игроков":
        st.header("Актуальный рейтинг игроков")
        stats = process_and_save(df_compile_stats)
        stats = rename_columns(stats)
        st.dataframe(stats)
    
    # 2. Статистика за сезон
    elif action == "Статистика игроков за сезон":
        st.header("Статистика игроков за сезон")
        # Здесь используется только один selectbox – пользователь может набрать нужные символы,
        # и в списке останутся только подходящие сезоны.
        season_id = st.selectbox("Выберите сезон", available_seasons)
        
        # Фильтруем DataFrame по сезону и исключаем amplua равное 8
        df_filtered = df_merged[(df_merged["ID season"] == season_id) & (df_merged["amplua"] != 8)]

        # Получаем уникальные идентификаторы для защитников (amplua == 9)
        players_9 = df_filtered[df_filtered["amplua"] == 9]["ID player"].unique()

        # Получаем уникальные идентификаторы для атакующих (amplua == 10)
        players_10 = df_filtered[df_filtered["amplua"] == 10]["ID player"].unique()

        st.write("Выберите игроков или оставьте поля пустыми:")

        players_input_9 = st.multiselect("Защитники", options=players_9, default=players_9[:2])
        players_input_10 = st.multiselect("Атакующие", options=players_10, default=players_10[:2])

        # Объединяем выбранные идентификаторы игроков
        players_input = list(players_input_9) + list(players_input_10)

        if st.button("Рассчитать статистику"):
            result_df = process_season(df_compile_stats, df_history, season_id, players_input)
            fig1, fig2, fig3, fig4, fig5 = plot_player_ratings(result_df, season_id)
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.pyplot(fig3)
            st.pyplot(fig4)
            st.pyplot(fig5)
            result_df = rename_columns(result_df)
            st.dataframe(result_df)
    
    # 3. Визуализация команд
    elif action == "Визуализация команд":
        st.header("Визуализация рейтингов команд")
        # Сезон можно выбирать аналогичным образом, или оставить пустым для всех сезонов
        season_id = st.selectbox("Выберите сезон (или оставьте пустым)", [""] + available_seasons)
        season_id = season_id if season_id != "" else None
        
        teams_in_season = df_merged[df_merged["ID season"] == season_id]["ID team"].unique()
        available_teams = sorted(teams_in_season)

        team_ids = st.multiselect("Выберите команды", available_teams, default=available_teams[:4])
        
        if st.button("Построить график"):
            df_team, fig_total, fig_stacked, fig_scatter, fig_metric, fig_radar = plot_team_ratings(df_compile_stats, df_history, season_id, team_ids)
            st.pyplot(fig_total)
            st.pyplot(fig_stacked)
            st.pyplot(fig_scatter)
            st.pyplot(fig_metric)
            st.pyplot(fig_radar)
            mapping = {
                'ID team': 'ID команды',
                'games': 'игры',
                'team_rating': 'рейтинг',
                'division': 'дивизион',
            }
            # Переименовываем столбцы, если они присутствуют в датафрейме
            df_team = df_team.rename(columns=mapping)
            st.dataframe(df_team)
