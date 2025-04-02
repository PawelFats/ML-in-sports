import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_player_stats(df, output_file=r"data\processed\red_method\player_stats.csv"):
    '''
    Считает суммарные достижения для каждого игрока за все время
    '''
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
    print(f"Файл сохранен: {output_file}")
    
    return player_stats

def calculate_points(df, coefficient, amplua):
    '''
    Считает очки игрока для каждого показателя учитывая его амплуа.
    '''
    df_filtered = df[df['amplua'] == amplua].copy()
    
    for col in ['goals', 'assists', 'throws_by', 'shot_on_target', 'blocked_throws', 'p_m']:
        df_filtered[f'p_{col}'] = ((df_filtered[col] + coefficient * df_filtered['games']) ** 2) / df_filtered['games']
    df_filtered['player_rating'] = df_filtered[['p_goals', 'p_assists', 'p_throws_by', 
                                                'p_shot_on_target', 'p_blocked_throws', 'p_m']].sum(axis=1)
    
    return round(df_filtered, 2)

def process_and_save(df, output_file=r"data\processed\red_method\player_stats_with_points.csv"):
    '''
    Рассчитывает и сохраняет актуальные рейтинги игрков.
    '''
    df_stats = calculate_player_stats(df)
    
    # Вычисляем очки для каждого амплуа с соответствующими коэффициентами
    df_defenders = calculate_points(df_stats, 2/3, 9)
    df_forwards = calculate_points(df_stats, 1/6, 10)
    
    # Объединяем результаты
    df_final = pd.concat([df_defenders, df_forwards])
    
    df_final = df_final.reindex(columns=['ID player', 'amplua', 'games', 'goals', 'p_goals', 'assists', 'p_assists',  'throws_by', 'p_throws_by', 'shot_on_target', 'p_shot_on_target', 
                                         'blocked_throws', 'p_blocked_throws', 'p_m', 'p_p_m', 'player_rating'])

    df_final.to_csv(output_file, index=False)
    print(f"Файл сохранен: {output_file}")
    
    return df_final

def process_season(compile_stats_path, game_history_path, season_id, player_ids=None, 
                   output_file=r"data\processed\red_method\season_player_stats_with_points.csv"):
    """
    Функция:
    1. Загружает данные из compile_stats.csv и game_history.csv.
    2. Фильтрует игры, оставляя только те, которые относятся к указанному сезону (по 'ID season').
    3. Если указан список player_ids, оставляет только данные для этих игроков.
    4. Выводит дополнительную информацию: число уникальных игр, число игроков по амплуа, число команд.
    5. Вызывает функцию process_and_save для расчёта статистики и рейтингов.
    """
    # Загружаем таблицу со статистикой игр игроков
    df_compile = pd.read_csv(compile_stats_path)
    
    # Загружаем историю игр с информацией о сезоне
    df_history = pd.read_csv(game_history_path, sep=";")
    
    # Фильтруем историю игр по нужному сезону
    df_history_season = df_history[df_history["ID season"] == season_id]
    
    # Оставляем только игры, присутствующие в истории выбранного сезона
    df_season = pd.merge(df_compile, df_history_season[["ID"]], left_on="ID game", right_on="ID", how="inner")
    
    # Если указан список номеров игроков, оставляем только их
    if player_ids is not None and len(player_ids) > 0:
        df_season = df_season[df_season["ID player"].isin(player_ids)]
    
    # Вывод дополнительной информации
    unique_games = df_season['ID game'].nunique()
    unique_teams = df_season['ID team'].nunique()
    unique_players_amplua = df_season.groupby('amplua')['ID player'].nunique()
    
    print(f"Уникальных игр в сезоне: {unique_games}")
    print(f"Уникальных команд в сезоне: {unique_teams}")
    print("Уникальных игроков по амплуа:")
    for amplua, count in unique_players_amplua.items():
        print(f"  Амплуа {amplua}: {count}")
    
    # Вызываем функцию, которая считает статистику и рассчитывает рейтинги с учетом сезона
    df_final = process_and_save(df_season, output_file=output_file)
    
    return df_final

def plot_player_ratings(result_df, season_id):
    '''
    Отрисовывает два графика. 
    Первый: сумарный рейтинг сегментируя столбец по определнным показателям.
    Второй: общее кол-во игр для рассматриваемых игрков.
    '''
    # Словарь для переименования метрик на русский
    metric_labels = {
        'p_goals': 'голы',
        'p_assists': 'ассисты',
        'p_throws_by': 'мимо',
        'p_shot_on_target': 'в створ',
        'p_blocked_throws': 'блокированные',
        'p_p_m': 'п/м'
    }
    
    # Приведение ID игрока к строковому типу для корректного отображения на оси X
    players = result_df['ID player'].astype(str)
    metrics = list(metric_labels.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # График 1: Составной столбчатый график (stacked bar chart) с русскими названиями показателей
    fig, ax = plt.subplots(figsize=(14,8))
    bottom = np.zeros(len(result_df))
    for i, metric in enumerate(metrics):
        ax.bar(players, result_df[metric], bottom=bottom, color=colors[i], label=metric_labels[metric])
        bottom += result_df[metric].values

    ax.set_xlabel('ID игрока', fontsize=12)
    ax.set_ylabel('Рейтинговые очки', fontsize=12)
    ax.set_title('Структура рейтингов по показателям за сезон {}'.format(season_id), fontsize=14)
    ax.legend(title='Показатели', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
        
    # График 2: График количества игр
    fig3, ax3 = plt.subplots(figsize=(14,6))
    ax3.bar(players, result_df['games'], color='skyblue')
    ax3.set_xlabel('ID игрока', fontsize=12)
    ax3.set_ylabel('Количество игр', fontsize=12)
    ax3.set_title('Количество игр за сезон {}'.format(season_id), fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_team_ratings(df_compile, df_history, season_id=None, team_ids=None):
    """
    Функция:
    1. Загружает данные из compile_stats.csv и game_history.csv.
    2. Если season_id передан, фильтрует игры по сезону.
    3. Если передан список team_ids, оставляет данные только по указанным командам.
    4. Группирует данные по 'ID player', 'amplua' и 'ID team', суммируя показатели игр и статистику.
    5. Рассчитывает рейтинговые очки для каждого игрока по тем же формулам:
         p_stat = ((stat + coefficient * games) ** 2) / games,
       где коэффициент равен 2/3 для амплуа 9 (защитники) и 1/6 для амплуа 10 (форварды).
    6. Суммирует рейтинги игроков по командам и строит столбчатую диаграмму суммарного рейтинга команд.
    """    
    # Если указан сезон, фильтруем по нему через историю игр
    if season_id is not None:
        df_history_season = df_history[df_history["ID season"] == season_id]
        # Фильтруем статистику, оставляя только игры, присутствующие в истории выбранного сезона
        df = pd.merge(df_compile, df_history_season[["ID", "division"]], left_on="ID game", right_on="ID", how="inner")
    else:
        df = pd.merge(
            df_compile,
            df_history[["ID", "division"]],
            left_on="ID game",
            right_on="ID",
            how="left"
        )
    
    # Для каждой команды оставляем одно значение division (например, первое)
    df_team_div = df.groupby('ID team', as_index=False)['division'].first()

    # Фильтруем по выбранным командам, если список передан
    if team_ids is not None and len(team_ids) > 0:
        df = df[df["ID team"].isin(team_ids)]
    
    # Группируем данные по игроку, амплуа и команде (на случай, если игроки выступали за одну команду)
    df_grouped = df.groupby(['ID player', 'amplua', 'ID team']).agg(
        games=('ID game', 'nunique'),
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        throws_by=('throws by', 'sum'),
        shot_on_target=('a shot on target', 'sum'),
        blocked_throws=('blocked throws', 'sum'),
        p_m=('p/m', 'sum')
    ).reset_index()
        
    # Вычисляем рейтинги для защитников и форвардов с разными коэффициентами
    df_def = df_grouped[df_grouped['amplua'] == 9]
    df_for = df_grouped[df_grouped['amplua'] == 10]
    
    df_def_calc = calculate_points(df_def, 2/3, 9)
    df_for_calc = calculate_points(df_for, 1/6, 10)
    
    # Объединяем результаты
    df_players = pd.concat([df_def_calc, df_for_calc])
    
    # Группируем по командам, суммируя рейтинги игроков
    df_team = df_players.groupby('ID team').agg(
        team_rating=('player_rating', 'sum')
    ).reset_index()
    
    df_team['team_rating'] = round(df_team['team_rating'], 2)
    
    # Объединяем итоговую таблицу команд с информацией о дивизионе
    df_team = pd.merge(df_team, df_team_div, on='ID team', how='left')

    # Построение графика
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df_team['ID team'].astype(str) + ' (Div ' + df_team['division'].astype(str) + ')', df_team['team_rating'], color='teal')
    if season_id is not None:
        plt.title(f'Суммарный рейтинг команд в сезоне {season_id}', fontsize=14)
    else:
        plt.title('Суммарный рейтинг команд за всё время', fontsize=14)
    plt.xlabel('ID команды', fontsize=12)
    plt.ylabel('Суммарный рейтинг', fontsize=12)
    plt.xticks(rotation=45)

    for bar, team_rating in zip(bars, df_team['team_rating']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, team_rating, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    
    return df_team

