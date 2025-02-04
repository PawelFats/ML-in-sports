import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from deap import base, creator, tools, algorithms
import random

# нормализации данных
def scale_and_select_features(input_file):
    # Считываем данные из CSV файла
    df = input_file

    # Определяем список столбцов, которые будут удалены перед масштабированием
    removed_columns = ["ID team", "ID opponent", "result", "ID game", "ID season", "stage", "division", "date"]
    
    # Выбираем столбцы, которые будут масштабироваться
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    # Создаем объект MinMaxScaler для нормализации данных
    scaler = MinMaxScaler()

    # Применяем масштабирование к выбранным столбцам
    df[selected_columns] = scaler.fit_transform(df[selected_columns])
    
    return df

def GetTeamStat(team_id, matches):
    team_matches = matches[(matches['ID team'] == team_id) | (matches['ID opponent'] == team_id)]
    
    total_wins = 0
    total_losses = 0
    total_draws = 0
    sum_columns = {
        'G': ['GT', 'GO'],
        'time': ['timeT', 'timeO'],
        'TB': ['TB_T', 'TB_O'],
        'Shot': ['ShotT', 'ShotO'],
        'BT': ['BT_T', 'BT_O'],
        'PM': ['PM_T', 'PM_O'],
        'As': ['As_T', 'As_O']
    }
    avg_columns = {
        'refl_sh': ['refl_sh_T', 'refl_sh_O'],
        'ELO': ['ELO', 'ELO_O'],
        'E_S': ['E_S', 'E_S_O'],
        'rating': ['rating_T', 'rating_O']
    }

    team_sums = {key: 0 for key in sum_columns}
    team_averages = {key: [] for key in avg_columns}

    for _, row in team_matches.iterrows():
        if row['ID team'] == team_id:
            if row['result'] == 1:
                total_wins += 1
            elif row['result'] == -1:
                total_losses += 1
            elif row['result'] == 0:
                total_draws += 1

            for sum_key, sum_cols in sum_columns.items():
                team_sums[sum_key] += sum(row[col] for col in sum_cols)
            
            for avg_key, avg_cols in avg_columns.items():
                team_averages[avg_key].extend(row[col] for col in avg_cols)
        else:
            if row['result'] == 1:
                total_losses += 1
            elif row['result'] == -1:
                total_wins += 1
            elif row['result'] == 0:
                total_draws += 1

            for sum_key, sum_cols in sum_columns.items():
                team_sums[sum_key] += sum(row[col] for col in sum_cols)
            
            for avg_key, avg_cols in avg_columns.items():
                team_averages[avg_key].extend(row[col] for col in avg_cols)

    team_avg_values = {key: sum(vals) / len(vals) for key, vals in team_averages.items() if vals}

    result_vector = [
        total_wins, total_losses, total_draws,
        *team_sums.values(),
        *team_avg_values.values()
    ]

    return result_vector

def GetAllTeamsStats(matches):
    team_ids = pd.concat([matches['ID team'], matches['ID opponent']]).unique()

    all_teams_stats = {}
    for team_id in team_ids:
        all_teams_stats[team_id] = GetTeamStat(team_id, matches)

    return all_teams_stats

def GetTrainingData(matches, cutoff_date):
    matches_train = matches[matches['date'] < cutoff_date]
    matches_test = matches[matches['date'] >= cutoff_date]

    all_teams_stats_train = GetAllTeamsStats(matches_train)
    all_teams_stats_test = GetAllTeamsStats(matches_test)

    totalNumGames_train = len(matches_train.index)
    totalNumGames_test = len(matches_test.index)
    numFeatures = len(next(iter(all_teams_stats_train.values())))  # длина вектора любой команды
    
    xTrain = np.zeros((totalNumGames_train, numFeatures))
    yTrain = np.zeros((totalNumGames_train))
    xTest = np.zeros((totalNumGames_test, numFeatures))
    yTest = np.zeros((totalNumGames_test))
    
    indexCounter_train = 0
    indexCounter_test = 0
    
    for _, row in matches_train.iterrows():
        team_id = row['ID team']
        opponent_id = row['ID opponent']
        result = row['result']

        team_vector = all_teams_stats_train[team_id]
        opponent_vector = all_teams_stats_train[opponent_id]

        if len(team_vector) != len(opponent_vector):
            print(f"Error: Vectors for team {team_id} and opponent {opponent_id} are of different lengths.")
            continue

        difference_vector = [team_value - opponent_value for team_value, opponent_value in zip(team_vector, opponent_vector)]

        xTrain[indexCounter_train] = difference_vector
        yTrain[indexCounter_train] = 1 if result == 1 else 0
        
        indexCounter_train += 1

    for _, row in matches_test.iterrows():
        team_id = row['ID team']
        opponent_id = row['ID opponent']
        result = row['result']

        team_vector = all_teams_stats_test[team_id]
        opponent_vector = all_teams_stats_test[opponent_id]

        if len(team_vector) != len(opponent_vector):
            print(f"Error: Vectors for team {team_id} and opponent {opponent_id} are of different lengths.")
            continue

        difference_vector = [team_value - opponent_value for team_value, opponent_value in zip(team_vector, opponent_vector)]

        xTest[indexCounter_test] = difference_vector
        yTest[indexCounter_test] = 1 if result == 1 else 0
        
        indexCounter_test += 1

    return xTrain, yTrain, xTest, yTest

# Функция для вывода вероятности победы команды
def get_team_win_probability(new_models, matches, team1_id, team2_id):
    team1_vector = GetTeamStat(team1_id, matches)
    team2_vector = GetTeamStat(team2_id, matches)
    difference_vector = [team1_value - team2_value for team1_value, team2_value in zip(team1_vector, team2_vector)]
    predicted_probability = new_models["LogisticRegression"].predict_proba([difference_vector])[:, 1]
    return predicted_probability[0]

# Преобразование вероятностей в проценты с помощью нормализации
def normalize_probabilities(probabilities):
    total_prob = sum(probabilities)
    if total_prob == 0:
        return [0] * len(probabilities)
    normalized_probs = [prob / total_prob for prob in probabilities]
    return normalized_probs

# Функция для получения уникальных команд
def get_unique_teams(matches):
    return pd.unique(matches[['ID team', 'ID opponent']].values.ravel('K'))

def get_team_win_probabilities(new_models, team_id, matches):
    unique_teams = get_unique_teams(matches)
    win_probabilities = {}
    for opp_team_id in unique_teams:
        if opp_team_id != team_id:
            win_probability_lr1 = get_team_win_probability(new_models, matches, team_id, opp_team_id)
            win_probability_lr2 = get_team_win_probability(new_models, matches, opp_team_id, team_id)
            # Получение вероятностей
            probabilities = [win_probability_lr1, win_probability_lr2]
            # Нормализация вероятностей
            normalized_probs = normalize_probabilities(probabilities)
            
            win_probabilities[opp_team_id] = normalized_probs[0]
    return win_probabilities


def process_season_data(season):
    game_stats = pd.read_csv("game_stats_one_r.csv")
    
    # Фильтрация строк с ID season < заданного значения
    game_before_season = game_stats[game_stats['ID season'] < season]

    # Фильтрация строк с ID season = заданному значению
    current_season_games = game_stats[game_stats['ID season'] == season]

    # Получение уникальных номеров команд и их дивизионов
    unique_teams = pd.concat([current_season_games[['ID team', 'division']],
                              current_season_games[['ID opponent', 'division']].rename(columns={'ID opponent': 'ID team'})]).drop_duplicates()
    
    # Извлечение уникальных номеров команд
    unique_team_ids = unique_teams['ID team'].unique()

    # Фильтрация строк в game_before_season с командами из unique_teams
    game_before_season = game_before_season[
        (game_before_season['ID team'].isin(unique_team_ids)) | 
        (game_before_season['ID opponent'].isin(unique_team_ids))
    ]
    
    # Проверка, все ли команды из unique_teams есть в game_before_season
    teams_in_before_season = set(game_before_season['ID team']).union(set(game_before_season['ID opponent']))
    teams_not_in_before_season = unique_teams[~unique_teams['ID team'].isin(teams_in_before_season)]

    # Удаление команд, которые не имеют игр в game_before_season_var из unique_teams
    unique_teams = unique_teams[unique_teams['ID team'].isin(teams_in_before_season)]

    # Сохранение game_before_season в переменную
    game_before_season_var = game_before_season.copy()

    return game_before_season_var, unique_teams

# Функция для получения вероятностей побед для всех уникальных матчей в каждом дивизионе
def simulate_all_matches(division_teams):
    match_results = []
    for division, teams in division_teams.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):  # Изменено для избегания дублирования матчей
                team1_id = teams[i]
                team2_id = teams[j]
                # Получение вероятности победы команды 1
                win_probability_lr1 = get_team_win_probability(team1_id, team2_id)
                win_probability_lr2 = get_team_win_probability(team2_id, team1_id)
                
                # Получение вероятностей
                probabilities = [win_probability_lr1, win_probability_lr2]
                
                # Нормализация вероятностей
                normalized_probs = normalize_probabilities(probabilities)
                    
                match_results.append({
                    'ID team': team1_id,
                    'ID opponent': team2_id,
                    'division': division,
                    '% team': normalized_probs[0] * 100,
                    '% opponent': normalized_probs[1] * 100
                })
    return pd.DataFrame(match_results)

# Функция для вычисления среднего значения наибольших вероятностей для каждого дивизиона
def calculate_average_highest_probabilities(match_df):
    division_probabilities = match_df.groupby('division').apply(
        lambda df: df[['% team', '% opponent']].max(axis=1).mean()
    )
    return division_probabilities

# Генетический алгоритм распределения, работает на данном этапе лучше предыдущих
def rank_teams(list_team, model_file, num_divisions, min_teams_per_division=3, num_generations=100, population_size=300):
    model = joblib.load(model_file)
    
    # Получение уникальных команд
    unique_teams = list_team['ID team'].unique()
    num_teams = len(unique_teams)
    if num_teams < num_divisions * min_teams_per_division:
        raise ValueError("Недостаточно команд для распределения по заданному количеству дивизионов")

    # Генерация всех возможных матчей
    all_matches = []
    for team1 in unique_teams:
        for team2 in unique_teams:
            if team1 != team2:
                win_prob1 = get_team_win_probability(team1, team2)
                win_prob2 = get_team_win_probability(team2, team1)
                probabilities = normalize_probabilities([win_prob1, win_prob2])
                all_matches.append([team1, team2, probabilities[0] * 100, probabilities[1] * 100])
    
    matches_df = pd.DataFrame(all_matches, columns=['ID team', 'ID opponent', '%T', '%O'])
    # Генетический алгоритм
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_teams), num_teams)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # Разбиваем индивид на дивизионы
        divisions = [individual[i::num_divisions] for i in range(num_divisions)]
        
        # Проверка на минимальное количество команд в каждом дивизионе
        if any(len(div) < min_teams_per_division for div in divisions):
            return float('inf'),
        # Проверка на уникальность команд
        flat_list = [item for sublist in divisions for item in sublist]
        if len(flat_list) != len(set(flat_list)):
            return float('inf'),
        
        score = 0
        for div in divisions:
            div_teams = unique_teams[div]
            div_matches = matches_df[(matches_df['ID team'].isin(div_teams)) & (matches_df['ID opponent'].isin(div_teams))]
            max_probs = div_matches[['%T', '%O']].max(axis=1)
            score += max_probs.mean()
        score /= num_divisions
        return score,

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, verbose=False)

    best_ind = tools.selBest(population, 1)[0]
    divisions = [best_ind[i::num_divisions] for i in range(num_divisions)]
    final_divisions = []
    for i, div in enumerate(divisions):
        for idx in div:
            final_divisions.append([unique_teams[idx], f'Division {i+1}'])
    
    final_df = pd.DataFrame(final_divisions, columns=['ID team', 'division'])
    
    # Сохранение финальной таблицы
    matches_df.to_csv('data/interim/matches_rangirov.csv', index=False)
    final_df.to_csv('team_rangirov.csv', index=False)
    #final_df.to_excel('xlsx.xlsx', index=False, float_format='%.2f')
    print(f"Количество уникальных команд: {len(final_df)}")