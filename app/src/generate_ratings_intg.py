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

#Функция посдчет для каждого игрока его срежних показателей
def calculate_mean_player_stats(compile_stats):
    # Фильтрация данных для игроков с amplua 9 и 10 (9 - нападающий, 10 - защитник)
    filtered_stats = compile_stats[(compile_stats['amplua'] == 9) | (compile_stats['amplua'] == 10)]
    
    # Преобразование времени на льду из секунд в минуты
    filtered_stats['total time on ice'] = filtered_stats['total time on ice'] / 60
    
    filtered_stats.rename(columns={
        'total time on ice': 'time',
        'goals': 'G',
        'assists': 'As',
        'throws by': 'TB',
        'a shot on target': 'Shot',
        'blocked throws': 'BT',
        'p/m': 'pm'
    }, inplace=True)
    
    # Группировка данных по командам и игрокам, а также рассчет средних значений
    player_team_stats = filtered_stats.groupby(['ID player','amplua']).agg({
        'time': 'mean',
        'G': 'mean',
        'As': 'mean',
        'TB': 'mean',
        'Shot': 'mean',
        'BT': 'mean',
        'pm': 'mean'
    }).reset_index()
    
    return player_team_stats
    
def calculate_mean_goalk_stats(goalk_data_game):
    
    goalk_data_game.rename(columns={
        'missed pucks': 'MisG',
        'total throws': 'TotalTr',
        '% of reflected shots': 'ReflSh'
    }, inplace=True)
    
    # Группировка данных по игрокам, а также рассчет средних значений
    goalk_data_game = goalk_data_game.groupby(['ID player']).agg({
        'MisG': 'mean',
        'TotalTr': 'mean',
        'ReflSh': 'mean'
    }).reset_index()
    
    return goalk_data_game

#расчет отклонений для всех полевых игрокв и создание таблицы
def calculate_and_add_deviations(df):
    
    overall_mean_amplua_10 = df[df['amplua'] == 10].mean()
    overall_mean_amplua_9 = df[df['amplua'] == 9].mean()
    # # Замена нулей на небольшое число для предотвращения деления на ноль
    overall_mean_amplua_9 = overall_mean_amplua_9.replace(0, 1e-6)
    overall_mean_amplua_10 = overall_mean_amplua_10.replace(0, 1e-6)
    df = df.replace(0, 1e-6)
    # Рассчитываем отклонения для амплуа 10
    deviations_amplua_10 = (df[df['amplua'] == 10].drop(['ID player', 'amplua'], axis=1) - overall_mean_amplua_10) / (np.abs(overall_mean_amplua_10)) * 100

    # Рассчитываем отклонения для амплуа 9
    deviations_amplua_9 = (df[df['amplua'] == 9].drop(['ID player', 'amplua'], axis=1) - overall_mean_amplua_9) / (np.abs(overall_mean_amplua_9)) * 100

    # Добавляем столбцы с отклонениями для амплуа 10
    for col in deviations_amplua_10.columns:
        if col != 'ID player' and col != 'amplua':
            deviation_col_name = f'dev_{col}'
            df.loc[df['amplua'] == 10, deviation_col_name] = deviations_amplua_10[col]

    # Добавляем столбцы с отклонениями для амплуа 9
    for col in deviations_amplua_9.columns:
        if col != 'ID player' and col != 'amplua':
            deviation_col_name = f'dev_{col}'
            df.loc[df['amplua'] == 9, deviation_col_name] = deviations_amplua_9[col]

    return df
#############################################################################################################################
#расчет отклонений для всех вратарей
def calculate_and_add_deviations_goalk(df):
    # Рассчитываем отклонения для амплуа 8
    overall_mean_amplua_8 = df.mean()
    
    # Рассчитываем отклонения и добавляем их в исходный DataFrame
    deviation = df.drop(['ID player'], axis=1)
    deviation = ((deviation - overall_mean_amplua_8) / overall_mean_amplua_8) * 100
    # Умножение столбца MisG на -1
    if 'MisG' in deviation.columns:
        deviation['MisG'] *= -1
    # Добавляем столбцы с отклонениями для амплуа 8
    for col in deviation.columns:
        if col != 'ID player':
            deviation_col_name = f'dev_{col}'
            df[deviation_col_name] = deviation[col]

    return df

#Формируем актуальные списки команд
def form_latest_teams(compile_stats_path, game_history_path, output_path):
    compile_stats = pd.read_csv(compile_stats_path)
    game_history = pd.read_csv(game_history_path, sep=';')
    
    #Добавление столбца 'date' в таблицу compile_stats
    compile_stats['date'] = compile_stats['ID game'].map(game_history.set_index('ID')['date'])
    
    # Преобразование столбца 'date' в формат datetime и сортировка данных по дате
    compile_stats['date'] = pd.to_datetime(compile_stats['date'])
    compile_stats_sorted = compile_stats.sort_values(by='date')

    # Нахождение последней игры для каждой команды
    last_game_per_team = compile_stats_sorted.groupby('ID team')['date'].max().reset_index()

    #Объединение таблиц для получения всех игроков, участвовавших в последних играх
    last_game_stats = pd.merge(last_game_per_team, compile_stats_sorted, on=['ID team', 'date'], how='left')

    #Создание итоговой таблицы с отдельными строками для каждой комбинации команды и игрока
    final_team_rosters = last_game_stats[['ID team', 'ID player']]
    ################################################################################
    # Добавление столбца "division" в итоговую таблицу
    final_team_rosters['division'] = final_team_rosters['ID team'].apply(lambda x: game_history.loc[(game_history['ID firstTeam'] == x) | (game_history['ID secondTeam'] == x), 'division'].iloc[-1])
    ###############################################################################
    
    final_team_rosters.to_csv(output_path, index=False)
    unique_teams_count = final_team_rosters['ID team'].nunique()
    print("Результаты работы фнукции form_latest_teams:")
    print("Количество уникальных команд:", unique_teams_count)


    return final_team_rosters

# #Домножение показателей deviation_ на веса для полевых игроков
def add_weights(mean_stats_deviat_path):
    #mean_stats_deviat = pd.read_csv(mean_stats_deviat_path)
    
    mean_stats_deviat = mean_stats_deviat_path
    
    # Веса для каждого показателя в зависимости от амплуа
    weights = {
        10: {'time': 0.35, 'G': 0.95, 'As': 0.85, 'TB': 0.4, 'Shot': 0.81, 'BT': 0.44, 'pm': 0.55},
        9: {'time': 0.4, 'G': 0.6, 'As': 0.8, 'TB': 0.35, 'Shot': 0.43, 'BT': 0.85, 'pm': 0.75}
    }

    #Рассчитываем взвешенное отклонение для каждого игрока
    for amplua, weight_dict in weights.items():
        # Выбираем игроков определенной амплуа
        players = mean_stats_deviat[mean_stats_deviat['amplua'] == amplua]
        for col, weight in weight_dict.items():
            deviation_col = f'dev_{col}'
            if deviation_col in mean_stats_deviat.columns:
                # Домножаем отклонение на соответствующий вес
                mean_stats_deviat.loc[players.index, deviation_col] *= weight
    return mean_stats_deviat

#Расчет срдне интегрально отклоннеия для игрока в целом и соритровка по команде
def integral_dev(mean_stats_deviat_path):

    mean_stats_deviat = mean_stats_deviat_path
    
    # Список столбцов для исключения из расчета интегрального отклонения
    exclude_columns = ['amplua', 'time', 'G', 'As', 'TB', 'Shot', 'BT', 'pm']

    #Вычисление интегрального отклонения от модельных величин для каждого игрока
    columns_to_include = mean_stats_deviat.drop(['ID player'] + exclude_columns, axis=1)
    mean_stats_deviat['integral_dev'] = columns_to_include.sum(axis=1) / columns_to_include.shape[1]
    
    exclude_columns = ['amplua', 'time', 'G', 'As', 'TB', 'Shot', 'BT', 'pm', 'dev_time', 'dev_G', 'dev_As', 'dev_TB', 'dev_Shot', 'dev_BT', 'dev_pm']
    
    mean_stats_deviat = mean_stats_deviat.drop(exclude_columns, axis=1)
    
    return mean_stats_deviat

#Расчет рейтинга команд за все время
def calculate_team_ratings_all_time(latest_teams_path, goalkeepers_data_path, players_data_path, output_path):

    latest_teams = pd.read_csv(latest_teams_path)
    goalkeepers_data = pd.read_csv(goalkeepers_data_path)
    players_data = pd.read_csv(players_data_path)
    
    #Отбор необходимых столбцов
    goalkeepers_ratings = goalkeepers_data[['ID player', 'integral_dev']]
    players_ratings = players_data[['ID player', 'integral_dev']]
    
    #Объединение таблиц рейтингов вратарей и полевых игроков
    all_ratings = pd.concat([goalkeepers_ratings, players_ratings]).reset_index(drop=True)
    
    #Сопоставление игроков с их рейтингами
    latest_teams_with_ratings = pd.merge(latest_teams, all_ratings, on='ID player', how='left')
    
    #Вычисление суммарного рейтинга для каждой команды
    team_ratings = latest_teams_with_ratings.groupby('ID team')['integral_dev'].sum().reset_index()
    team_ratings = team_ratings.rename(columns={'integral_dev': 'PL_RATING'})
    
    team_ratings.to_csv(output_path, index=False, float_format='%.2f')
    
    return team_ratings

#Рейтинги игрокв окончательные
def pl_rating_info(latest_teams_path, goalkeepers_data_path, players_data_path, output_path):
    latest_teams = pd.read_csv(latest_teams_path)
    goalkeepers_data = pd.read_csv(goalkeepers_data_path)
    players_data = pd.read_csv(players_data_path)
    
    # Отбор необходимых столбцов и агрегация рейтингов
    goalkeepers_ratings = goalkeepers_data[['ID player', 'integral_dev']].rename(columns={'integral_dev': 'player_rating'})
    players_ratings = players_data[['ID player', 'integral_dev']].rename(columns={'integral_dev': 'player_rating'})

    # Объединение данных по ID player
    latest_teams = pd.merge(latest_teams, goalkeepers_ratings, on='ID player', how='left')
    latest_teams = pd.merge(latest_teams, players_ratings, on='ID player', how='left')
    
    # Обработка возможных отсутствующих значений
    latest_teams['division'] = latest_teams['division'].fillna(0)

    # Объединение рейтингов в один столбец
    latest_teams['player_rating'] = latest_teams['player_rating_x'].fillna(latest_teams['player_rating_y'])
    latest_teams.drop(['player_rating_x', 'player_rating_y'], axis=1, inplace=True)
    
    # Добавление рейтинга в зависимости от дивизии
    latest_teams['player_rating'] += latest_teams['division'].map({0: 500, 1: 400, 2: 300, 3: 200, 4: 100, 5: 0})
    
    # Сортировка по дивизиону и рейтингу
    latest_teams = latest_teams.sort_values(by=['division', 'player_rating'], ascending=[True, False])
        
    latest_teams.to_csv(output_path, index=False, float_format='%.2f')
    
    return latest_teams

#Рейтинги игрокв окончательные
def pl_rating_info(latest_teams_path, goalkeepers_data_path, players_data_path, output_path):
    latest_teams = pd.read_csv(latest_teams_path)
    goalkeepers_data = pd.read_csv(goalkeepers_data_path)
    players_data = pd.read_csv(players_data_path)
    
    # Отбор необходимых столбцов и агрегация рейтингов
    goalkeepers_ratings = goalkeepers_data[['ID player', 'integral_dev']].rename(columns={'integral_dev': 'player_rating'})
    players_ratings = players_data[['ID player', 'integral_dev']].rename(columns={'integral_dev': 'player_rating'})

    # Объединение данных по ID player
    latest_teams = pd.merge(latest_teams, goalkeepers_ratings, on='ID player', how='left')
    latest_teams = pd.merge(latest_teams, players_ratings, on='ID player', how='left')
    
    # Обработка возможных отсутствующих значений
    latest_teams['division'] = latest_teams['division'].fillna(0)

    # Объединение рейтингов в один столбец
    latest_teams['player_rating'] = latest_teams['player_rating_x'].fillna(latest_teams['player_rating_y'])
    latest_teams.drop(['player_rating_x', 'player_rating_y'], axis=1, inplace=True)
    
    # Добавление рейтинга в зависимости от дивизии
    latest_teams['player_rating'] += latest_teams['division'].map({0: 500, 1: 400, 2: 300, 3: 200, 4: 100, 5: 0})
    
    # Сортировка по дивизиону и рейтингу
    latest_teams = latest_teams.sort_values(by=['division', 'player_rating'], ascending=[True, False])
        
    latest_teams.to_csv(output_path, index=False, float_format='%.2f')
    
    return latest_teams

def calculate_team_ratings(pl_rating_info_path, output_path):

    pl_rating_info = pd.read_csv(pl_rating_info_path)
    
    # Группировка данных по командам и суммирование рейтингов игроков
    team_ratings = pl_rating_info.groupby('ID team')['player_rating'].sum().reset_index()
    team_ratings = team_ratings.rename(columns={'player_rating': 'PL_RATING'})
    team_ratings = team_ratings.sort_values(by=['PL_RATING'], ascending=[False])
    # Сохранение итоговой таблицы с рейтингами команд
    team_ratings.to_csv(output_path, index=False, float_format='%.2f')
    
    return team_ratings

#Расчет рейтингов за все игры
def calculate_ratings(game_stats_file, game_history_file, goalkeepers_file, output_file, team_output_file):
    compile_stats = pd.read_csv(game_stats_file)
    game_history = pd.read_csv(game_history_file)
    goalkeepers_data = pd.read_csv(goalkeepers_file)

    # Добавление столбца 'date' в таблицу compile_stats
    compile_stats['date'] = compile_stats['ID game'].map(game_history.set_index('ID')['date'])
    #compile_stats['date'] = pd.to_datetime(compile_stats['date'])

    # Вызов функции и вывод общих статистик для игроков
    mean_stats_pl = calculate_mean_player_stats(compile_stats)
    overall_stats_amplua_10, overall_stats_amplua_9 = calculate_overall_stats(mean_stats_pl)

    # Расчет средних значений для игроков и вратарей
    mean_stats_goalk = calculate_mean_goalk_stats(goalkeepers_data)
    overall_stats_amplua_8 = mean_stats_goalk.mean()

    # Сортировка игр по дате
    compile_stats_sorted = compile_stats.sort_values(by=['date', 'ID game'])

    # Итоговая таблица рейтингов
    final_ratings = []

    # Инициализация словаря для хранения количества игр для каждого игрока
    player_games_count = {}
    current_data = pd.DataFrame()

    # Проход по каждой игре
    for current_game in compile_stats_sorted['ID game'].unique():
        #print(current_game)
        # инфа о теккущей игре для которой расчет
        current_game_index = compile_stats[compile_stats['ID game'] == current_game] 
        
        # Данные по текущую игру
        current_data = pd.concat([current_data, current_game_index], ignore_index=True)

        # Подсчет количества появлений каждого игрока в текущей игре
        player_counts = current_game_index['ID player'].value_counts()
        # Обновление общего количества игр для каждого игрока
        for player, count in player_counts.items():
            if player in player_games_count:
                player_games_count[player] += count
            else:
                player_games_count[player] = count

        # Расчет средних значений для каждого игрока для текущей итерации
        mean_stats_pl = calculate_mean_player_stats(current_data)
        mean_stats_pl_with_deviations = calculate_and_add_deviations(mean_stats_pl)
        # Домножение показателей deviation_ на веса
        weighted_stats = add_weights(mean_stats_pl_with_deviations)
        # Расчет интегрального отклонения
        integral_ratings = integral_dev(weighted_stats)
        
        # Обработка вратарей для текущей итерации
        available_games = current_data['ID game'].unique()
        goalkeepers_data_current = goalkeepers_data[goalkeepers_data['ID game'].isin(available_games)]
        mean_stats_goalk_current = calculate_mean_goalk_stats(goalkeepers_data_current)
        mean_stats_goalk_with_deviations = calculate_and_add_deviations_goalk(mean_stats_goalk_current)
        mean_stats_goalk_with_deviations['integral_dev'] = mean_stats_goalk_with_deviations['dev_ReflSh'] * 15
        integral_goalk_ratings = mean_stats_goalk_with_deviations.drop(['MisG', 'TotalTr', 'ReflSh', 'dev_MisG', 'dev_ReflSh', 'dev_TotalTr'], axis=1)
        
        # Формирование актуальных составов команд
        current_teams = current_data[current_data['ID game'] == current_game][['ID team', 'ID player']]

        # Добавление данных в итоговую таблицу для игроков
        for _, row in current_teams.iterrows():
            team_id = row['ID team']
            player_id = row['ID player']
            
            # Проверяем, существуют ли данные для данного игрока в integral_ratings
            player_data = integral_ratings[integral_ratings['ID player'] == player_id]
            
            if not player_data.empty:  # Если данные существуют
                player_rating = player_data['integral_dev'].values[0]
                if player_games_count[player_id] < 10:
                    player_rating = 0.001
                final_ratings.append([current_game, team_id, player_id, player_rating])

        # Добавление данных в итоговую таблицу для вратарей
        current_goalkeepers = goalkeepers_data_current[goalkeepers_data_current['ID game'] == current_game][['ID team', 'ID player']].drop_duplicates()
        for _, row in current_goalkeepers.iterrows():
            team_id = row['ID team']
            player_id = row['ID player']
            
            # Проверяем, существуют ли данные для данного игрока в integral_goalk_ratings
            goalkeeper_data = integral_goalk_ratings[integral_goalk_ratings['ID player'] == player_id]
            
            if not goalkeeper_data.empty:  # Если данные существуют
                goalkeeper_rating = goalkeeper_data['integral_dev'].values[0]
                if player_games_count[player_id] < 5:
                    goalkeeper_rating = 0.001
                final_ratings.append([current_game, team_id, player_id, goalkeeper_rating])
    
    # Создание DataFrame из итоговой таблицы
    final_ratings_df = pd.DataFrame(final_ratings, columns=['ID game', 'ID team', 'ID player', 'pl_rating'])
    final_ratings_df.to_csv(output_file, index=False, float_format='%.2f')

    # Создание рейтинга команд
    # Группировка данных по 'ID game' и 'ID team' и суммирование рейтингов игроков
    team_ratings = final_ratings_df.groupby(['ID game', 'ID team'])['pl_rating'].sum().reset_index()
    # Переименование столбца для итоговой таблицы
    team_ratings.rename(columns={'pl_rating': 'rating'}, inplace=True)

    team_ratings['date'] = team_ratings['ID game'].map(game_history.set_index('ID')['date'])
    team_ratings['date'] = pd.to_datetime(team_ratings['date'])
    # Сортировка игр по дате
    team_ratings_sorted = team_ratings.sort_values(by=['date', 'ID game'])

    team_ratings_sorted.to_csv(team_output_file, index=False, float_format='%.2f')

#дОБАВЛЕНИЕ СТРОКИ old_rating
def add_old_ratings_to_teams(input_file, output_file):
    # Чтение данных из входного CSV файла
    df = pd.read_csv(input_file)

    # Инициализация словаря для хранения предыдущих рейтингов команд
    previous_ratings = {}

    # Список для хранения значений old_rating
    old_ratings = []

    # Итерация по строкам таблицы
    for index, row in df.iterrows():
        team_id = row['ID team']
        current_rating = row['rating']
        
        # Добавление старого рейтинга в список old_ratings
        if team_id in previous_ratings:
            old_ratings.append(previous_ratings[team_id])
        else:
            old_ratings.append(0)  # Если команды нет в списке, пишем 0
        
        # Обновление предыдущего рейтинга команды текущим рейтингом
        previous_ratings[team_id] = current_rating

    # Добавление нового столбца old_rating в DataFrame
    df['old_rating'] = old_ratings

    # Сохранение DataFrame в выходной CSV файл
    df.to_csv(output_file, index=False)


#Добавление интегрального рейтинга команд
def add_ratings_to_game_stats(ratings_file, game_stats_file, output_file):
    # Чтение данных из файлов
    ratings_df = pd.read_csv(ratings_file)
    game_stats_df = pd.read_csv(game_stats_file)

    # Добавление новых столбцов для рейтингов
    game_stats_df['rating_T'] = None
    game_stats_df['old_rating_T'] = None
    game_stats_df['rating_O'] = None
    game_stats_df['old_rating_O'] = None

    # Итерация по строкам game_stats_df и добавление данных рейтингов
    for index, row in game_stats_df.iterrows():
        game_id = row['ID game']
        team_id = row['ID team']
        opponent_id = row['ID opponent']

        # Поиск совпадений в таблице рейтингов
        team_rating = ratings_df[(ratings_df['ID game'] == game_id) & (ratings_df['ID team'] == team_id)]
        opponent_rating = ratings_df[(ratings_df['ID game'] == game_id) & (ratings_df['ID team'] == opponent_id)]

        # Если нашли совпадение по ID team
        if not team_rating.empty:
            game_stats_df.at[index, 'old_rating_T'] = team_rating.iloc[0]['old_rating']
            game_stats_df.at[index, 'rating_T'] = team_rating.iloc[0]['rating']

        # Если нашли совпадение по ID opponent
        if not opponent_rating.empty:
            game_stats_df.at[index, 'old_rating_O'] = opponent_rating.iloc[0]['old_rating']
            game_stats_df.at[index, 'rating_O'] = opponent_rating.iloc[0]['rating']

    # Сохранение обновленного DataFrame в новый CSV файл
    game_stats_df.to_csv(output_file, index=False)

    # Вывод первых 5 строк обновленного DataFrame
    print(game_stats_df.head())

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
    
    if st.button("🔁 Перерасчитать рейтинги"):
        with st.spinner("Выполняется перерасчет рейтингов..."):
            try:
                # #Расчет рейтингов за все игры
                # calculate_ratings(
                #     game_stats_file='data/targeted/compile_stats.csv', 
                #     game_history_file='data/raw/game_history.csv',
                #     goalkeepers_file='data/targeted/goalkeepers_data.csv', 
                #     output_file='data/processed/all/final_ratings.csv', 
                #     team_output_file='data/processed/all/RATING_TEAM_ALL_TIME.csv'
                #     )
                
                # Шаг 1: Пересчитываем статистику вратарей
                goalk_data_games = pd.read_csv('data/targeted/goalkeepers_data.csv')
                mean_stats_goalk = calculate_mean_goalk_stats(goalk_data_games)
                mean_stats_goalk.to_csv('data/processed/rating_last_time/mean_stats_goalk.csv', index=False, float_format='%.2f')

                # Шаг 2: Пересчитываем статистику полевых игроков
                compile_stats = pd.read_csv('data/targeted/compile_stats.csv')
                mean_stats_pl = calculate_mean_player_stats(compile_stats)
                mean_stats_pl.to_csv('data/processed/rating_last_time/mean_stats_pl.csv', index=False, float_format='%.2f')

                # add_old_ratings_to_teams(
                #     input_file='data/processed/all/RATING_TEAM_ALL_TIME.csv', 
                #     output_file='data/processed/all/RATING_TEAM_ALL_TIME_with_old_ratings.csv'
                # )

                # add_ratings_to_game_stats(
                #     'data/processed/all/RATING_TEAM_ALL_TIME_with_old_ratings.csv',
                #     'data/targeted/game_stats_one_r.csv',
                #     'data/targeted/game_stats_one_r.csv'
                # )

                st.success("Рейтинги успешно пересчитаны!")
            except Exception as e:
                st.error(f"Ошибка при перерасчёте рейтингов: {e}")

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

