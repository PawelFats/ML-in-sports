import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        #time=('total time on ice', 'sum')
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
                                                'p_shot_on_target', 'p_blocked_throws', 'p_p_m']].sum(axis=1)
    
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
    Отрисовывает несколько графиков:
    1. Составной столбчатый график (stacked bar chart) рейтингов по показателям.
    2. График количества игр.
    3. Тепловая карта показателей.
    4. Группированный бар-чарт по метрикам.
    5. Радарная диаграмма (спайдер-чарт) для топ-5 игроков по суммарному рейтингу.
    '''
    # Словарь для переименования метрик на русский
    metric_labels = {
        #'p_time': 'время', не учитывае вермя, его нужно перерассчитывать
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']#'#e377c2'

    # Рассчитываем процентное соотношение каждого показателя
    result_df_percent = result_df.copy()

    for metric in metrics:
        result_df_percent[metric] = (result_df_percent[metric] / result_df_percent['player_rating']) * 100
    
    # 1. График: Составной столбчатый график (stacked bar chart) рейтингов в процентах
    fig, ax = plt.subplots(figsize=(14,8))
    bottom = np.zeros(len(result_df_percent))
    for i, metric in enumerate(metrics):
        ax.bar(players, result_df_percent[metric], bottom=bottom, color=colors[i], label=metric_labels[metric])
        bottom += result_df_percent[metric].values
        
    ax.set_xlabel('ID игрока', fontsize=12)
    ax.set_ylabel('Процентное соотношение', fontsize=12)
    ax.set_title('Структура рейтингов по показателям в процентах за сезон {}'.format(season_id), fontsize=14)
    ax.legend(title='Показатели', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2. График: Количество игр
    fig2, ax2 = plt.subplots(figsize=(14,6))
    ax2.bar(players, result_df['games'], color='skyblue')
    ax2.set_xlabel('ID игрока', fontsize=12)
    ax2.set_ylabel('Количество игр', fontsize=12)
    ax2.set_title('Количество игр за сезон {}'.format(season_id), fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Тепловая карта показателей
    plt.figure(figsize=(14, 10))
    heatmap_data = result_df.set_index('ID player')[metrics].rename(columns=metric_labels)
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".1f", 
        cmap="coolwarm", 
        linewidths=0.5, 
        linecolor="white",
        cbar_kws={'label': 'Рейтинговые очки'}
    )
    plt.title(f'Тепловая карта показателей за сезон {season_id}\n', fontsize=16, pad=20)
    plt.xlabel('Показатели', fontsize=12)
    plt.ylabel('ID игрока', fontsize=12)
    plt.xticks(ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 4. Группированный бар-чарт по метрикам
    plt.figure(figsize=(16, 8))
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
        edgecolor='w'
    )
    plt.title(f'Распределение показателей по игрокам (сезон {season_id})', fontsize=16)
    plt.xlabel('ID игрока', fontsize=12)
    plt.ylabel('Рейтинговые очки', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Показатели', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. Радарная диаграмма для топ-5 игроков
    # Вычисляем суммарный рейтинг
    #result_df['total_rating'] = result_df[metrics].sum(axis=1)
    top_players = result_df.nlargest(5, 'player_rating')
    
    # Подготовка данных
    labels = list(metric_labels.values())
    num_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    # Стиль
    radar_colors = sns.color_palette("husl", 5)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Для каждого игрока
    for i, (_, row) in enumerate(top_players.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, color=radar_colors[i], linewidth=2, 
                label=f"ID {row['ID player']} (Σ={row['player_rating']:.1f})")
        ax.fill(angles, values, color=radar_colors[i], alpha=0.1)
    
    # Настройка визуала
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)
    plt.xticks(angles[:-1], labels, fontsize=10)
    plt.yticks(fontsize=8)
    plt.title('Топ-5 игроков по суммарному рейтингу\n', fontsize=16, pad=40)
    plt.legend(
        loc='upper right', 
        bbox_to_anchor=(1.4, 1.1),
        fontsize=10,
        frameon=True,
        shadow=True
    )
    
    # Линии сетки
    ax.spines['polar'].set_visible(False)
    ax.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()

def calc_team_rating(df_compile, df_history, season_id=None, team_ids=None):
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

    return df_team

def plot_team_ratings(df_team, season_id):
    """
    Функция:
    Суммирует рейтинги игроков по командам и строит столбчатую диаграмму суммарного рейтинга команд.
    """    
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

