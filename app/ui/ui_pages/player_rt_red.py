import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------
# Загрузка данных
# --------------------------
@st.cache_data
def load_data():
    df_history = pd.read_csv(r"C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\raw\game_history.csv", sep=";")
    df_compile_stats = pd.read_csv(r'C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\targeted\compile_stats.csv')
    df_goalk_stats = pd.read_csv(r'C:\Users\optem\Desktop\Magistracy\Диссертация\ML-in-sports\data\targeted\goalkeepers_data.csv')
    return df_history, df_compile_stats, df_goalk_stats

# --------------------------
# Расчёт статистики игроков
# --------------------------
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
        total_time=('total time on ice', 'sum')
    ).reset_index()

    player_stats.to_csv(output_file, index=False)
    
    return player_stats

def calculate_points(df, coefficient, amplua):
    """
    Считает очки игрока для каждого показателя с учетом его амплуа.
    """
    df_filtered = df[df['amplua'] == amplua].copy()
    
    for col in ['goals', 'assists', 'throws_by', 'shot_on_target', 'blocked_throws', 'p_m']:
        df_filtered[f'p_{col}'] = ((df_filtered[col] + coefficient * df_filtered['games']) ** 2) / df_filtered['games']
    df_filtered['player_rating'] = df_filtered[['p_goals', 'p_assists', 'p_throws_by', 
                                                 'p_shot_on_target', 'p_blocked_throws', 'p_m']].sum(axis=1)
    
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

# --------------------------
# Визуализация
# --------------------------
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
    
    # График 1: составной столбчатый график
    fig1, ax1 = plt.subplots(figsize=(14,8))
    bottom = np.zeros(len(result_df))
    for i, metric in enumerate(metrics):
        ax1.bar(players, result_df[metric], bottom=bottom, color=colors[i], label=metric_labels[metric])
        bottom += result_df[metric].values
    ax1.set_xlabel('ID игрока')
    ax1.set_ylabel('Рейтинговые очки')
    ax1.set_title(f'Структура рейтингов по показателям за сезон {season_id}')
    ax1.legend(title='Показатели')
    plt.xticks(rotation=45)
    
    # График 2: количество игр
    fig2, ax2 = plt.subplots(figsize=(14,6))
    ax2.bar(players, result_df['games'], color='skyblue')
    ax2.set_xlabel('ID игрока')
    ax2.set_ylabel('Количество игр')
    ax2.set_title(f'Количество игр за сезон {season_id}')
    plt.xticks(rotation=45)
    
    return fig1, fig2

def plot_team_ratings(df_compile, df_history, season_id=None, team_ids=None):
    """
    Функция:
      1. Загружает данные.
      2. При наличии season_id — фильтрует данные по сезону.
      3. При наличии team_ids — оставляет данные только для указанных команд.
      4. Группирует данные по игрокам, рассчитывает рейтинговые очки.
      5. Группирует по командам и строит столбчатую диаграмму суммарного рейтинга.
    """
    
    if season_id is not None:
        df_history_season = df_history[df_history["ID season"] == int(season_id)]
        df = pd.merge(df_compile, df_history_season[["ID", "division"]], left_on="ID game", right_on="ID", how="inner")
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
    
    if team_ids is not None and len(team_ids) > 0:
        df = df[df["ID team"].isin(team_ids)]
    
    df_grouped = df.groupby(['ID player', 'amplua', 'ID team']).agg(
        games=('ID game', 'nunique'),
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        throws_by=('throws by', 'sum'),
        shot_on_target=('a shot on target', 'sum'),
        blocked_throws=('blocked throws', 'sum'),
        p_m=('p/m', 'sum')
    ).reset_index()
        
    df_def = df_grouped[df_grouped['amplua'] == 9]
    df_for = df_grouped[df_grouped['amplua'] == 10]
    
    df_def_calc = calculate_points(df_def, 2/3, 9)
    df_for_calc = calculate_points(df_for, 1/6, 10)
    
    df_players = pd.concat([df_def_calc, df_for_calc])
    
    df_team = df_players.groupby('ID team').agg(
        team_rating=('player_rating', 'sum')
    ).reset_index()
    
    df_team['team_rating'] = round(df_team['team_rating'], 2)
    
    df_team = pd.merge(df_team, df_team_div, on='ID team', how='left')
    
    # Построение графика
    fig, ax = plt.subplots(figsize=(12, 7))
    team_labels = df_team['ID team'].astype(str) + ' (Div ' + df_team['division'].astype(str) + ')'
    bars = ax.bar(team_labels, df_team['team_rating'], color='teal')
    if season_id is not None:
        ax.set_title(f'Суммарный рейтинг команд в сезоне {season_id}', fontsize=14)
    else:
        ax.set_title('Суммарный рейтинг команд за всё время', fontsize=14)
    ax.set_xlabel('ID команды', fontsize=12)
    ax.set_ylabel('Суммарный рейтинг', fontsize=12)
    plt.xticks(rotation=45)
    for bar, team_rating in zip(bars, df_team['team_rating']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, team_rating, ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    
    return df_team, fig


def player_rt_red():
    st.title("Интерактивная визуализация статистики игроков")
    st.sidebar.header("Настройки")
    
    with st.spinner("Загрузка данных..."):
        df_history, df_compile_stats, df_goalk_stats = load_data()
    
    # Объединяем данные, чтобы брать только те сезоны, в которых есть игры из compile_stats
    df_merged = pd.merge(df_compile_stats, df_history, left_on="ID game", right_on="ID", how="inner")
    available_seasons = sorted(df_merged["ID season"].unique())
    
    # Если требуется, список доступных команд можно получить так:
    available_teams = sorted(df_compile_stats["ID team"].unique())
    
    available_players = sorted(df_compile_stats["ID player"].unique())
    
    action = st.sidebar.selectbox("Выберите действие", 
                                  ["Актуальный рейтинг игроков", "Статистика игроков за сезон", "Визуализация команд"])
    
    # 1. Общая статистика игроков
    if action == "Актуальный рейтинг игроков":
        st.header("Актуальный рейтинг игроков")
        stats = process_and_save(df_compile_stats)
        st.dataframe(stats)
    
    # 2. Статистика за сезон
    elif action == "Статистика игроков за сезон":
        st.header("Статистика игроков за сезон")
        # Здесь используется только один selectbox – пользователь может набрать нужные символы,
        # и в списке останутся только подходящие сезоны.
        season_id = st.selectbox("Выберите сезон", available_seasons)
        
        players_input = st.multiselect("Введите ID игроков (через запятую) или оставьте пустым", options=available_players, default=available_players[:4])
        
        # if players_input:
        #     try:
        #         player_ids = [int(pid.strip()) for pid in players_input.split(",") if pid.strip().isdigit()]
        #     except Exception as e:
        #         st.error("Ошибка при вводе ID игроков. Убедитесь, что вводите числа через запятую.")
        #         player_ids = None
        # else:
        #     player_ids = None

        if st.button("Рассчитать статистику"):
            result_df = process_season(df_compile_stats, df_history, season_id, players_input)
            fig1, fig2 = plot_player_ratings(result_df, season_id)
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.dataframe(result_df)
    
    # 3. Визуализация команд
    elif action == "Визуализация команд":
        st.header("Визуализация рейтингов команд")
        # Сезон можно выбирать аналогичным образом, или оставить пустым для всех сезонов
        season_id = st.selectbox("Выберите сезон (или оставьте пустым)", [""] + available_seasons)
        season_id = season_id if season_id != "" else None
        
        team_ids = st.multiselect("Выберите команды", available_teams)
        
        if st.button("Построить график"):
            df_team, fig = plot_team_ratings(df_compile_stats, df_history, season_id, team_ids)
            st.pyplot(fig)
            st.dataframe(df_team)


#df_team, fig = plot_team_ratings(df_compile_stats, df_history, season_id, team_ids)
