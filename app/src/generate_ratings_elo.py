import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def crt_game_stats(compile_stats, goalk_df, game_history_df):
    
    # Группировка данных по ID игры
    grouped_stats = compile_stats.groupby('ID game')
    
    # Создание списка для хранения данных о каждой игре
    game_data = []
    
    # Итерация по каждой группе данных (каждой игре)
    for game_id, group in grouped_stats:
        # Разделение данных на информацию о первой и второй команде
        team1_stats = group[group['ID team'] == group['ID team'].iloc[0]]
        team2_stats = group[group['ID team'] != group['ID team'].iloc[0]]
        
        # Информация о первой команде
        team1_info = {
            'ID team': team1_stats['ID team'].iloc[0],
            'GT': team1_stats['goals'].sum(),
            'timeT': team1_stats['total time on ice'].sum(),
            'TB_T': team1_stats['throws by'].sum(),
            'ShotT': team1_stats['a shot on target'].sum(),
            'BT_T': team1_stats['blocked throws'].sum(),
            'PM_T': team1_stats['p/m'].sum(),
            'As_T': team1_stats['assists'].sum()
        }
        
        # Информация о второй команде
        team2_info = {
            'GO': team2_stats['goals'].sum(),
            'timeO': team2_stats['total time on ice'].sum(),
            'TB_O': team2_stats['throws by'].sum(),
            'ShotO': team2_stats['a shot on target'].sum(),
            'BT_O': team2_stats['blocked throws'].sum(),
            'PM_O': team2_stats['p/m'].sum(),
            'As_O': team2_stats['assists'].sum()
        }
        
        # Результат игры
        if team1_info['GT'] > team2_info['GO']:
            result = 'W'  # Победа первой команды
        elif team1_info['GT'] < team2_info['GO']:
            result = 'L'  # Победа второй команды
        else:
            result = 'D'  # Ничья
        
        # Добавление информации о игре в список
        game_data.append({
            'ID game': game_id,
            'ID team': team1_info['ID team'],
            'ID opponent': team2_stats['ID team'].iloc[0],
            'result': result,
            **team1_info,
            **team2_info
        })
    
    # Создание DataFrame на основе списка данных об играх
    game_stats = pd.DataFrame(game_data)
    
    goalk = goalk_df
    game_history = game_history_df
    
    # Создание словаря, где ключами являются кортежи ('ID game', 'ID team'), а значениями - '% of reflected shots' из таблицы goalk
    goalk_dictT = goalk.set_index(['ID game', 'ID team'])['% of reflected shots'].to_dict()
    goalk_dictO = goalk.set_index(['ID game', 'ID team'])['% of reflected shots'].to_dict()
    
    # Добавление столбца '% goalk' в таблицу game_stats на основе соответствия 'ID game' и 'ID team'
    game_stats['refl_sh_T'] = game_stats.apply(lambda row: goalk_dictT.get((row['ID game'], row['ID team']), None), axis=1)
    game_stats['refl_sh_O'] = game_stats.apply(lambda row: goalk_dictO.get((row['ID game'], row['ID opponent']), None), axis=1)
    
    game_stats['ID season'] = game_stats['ID game'].map(game_history.set_index('ID')['ID season'])
    game_stats['stage'] = game_stats['ID game'].map(game_history.set_index('ID')['stage'])
    game_stats['division'] = game_stats['ID game'].map(game_history.set_index('ID')['division'])
    game_stats['date'] = game_stats['ID game'].map(game_history.set_index('ID')['date'])
    
    game_stats['date'] = pd.to_datetime(game_stats['date'])

    game_stats = game_stats.sort_values(by=['date', 'ID game'], ascending=[True, True])
    
    game_stats['ELO'] = ''
    game_stats['ELO_O'] = ''
    game_stats['E_S'] = ''
    game_stats['E_S_O'] = ''
    game_stats['ELO_old'] = ''
    game_stats['ELO_O_old'] = ''
    
    game_stats.to_csv('data/targeted/game_stats_one_r.csv', index=False)


def calculate_elo(game_stats_file_path, k_factor):
    K_FACTOR = k_factor#30

    game_stats = pd.read_csv(game_stats_file_path)

    # Сортировка данных по дате и ID game
    game_stats['date'] = pd.to_datetime(game_stats['date'])
    game_stats = game_stats.sort_values(by=['date', 'ID game'])

    # Словарь для хранения рейтингов команд
    elo_ratings = {}
    
    # Проход по каждой строке в отсортированных данных
    for index, row in game_stats.iterrows():
        # Получаем информацию о матче
        game_id = row['ID game']
        team_id = row['ID team']
        opponent_id = row['ID opponent']
        result = row['result']
        division = row['division']
        E_S= row['E_S']
        E_S_O = row['E_S_O']
            

        # Проверяем, есть ли уже рейтинг для команды, если нет, присваиваем начальный
        if team_id not in elo_ratings:
            initial_elo = 1500 - (division * 100) if division > 0 else 1500
            elo_ratings[team_id] = initial_elo
        if opponent_id not in elo_ratings:
            initial_elo = 1500 - (division * 100) if division > 0 else 1500
            elo_ratings[opponent_id] = initial_elo
    
        # Получаем текущий ELO рейтинг для команды и соперника
        team_elo = elo_ratings[team_id]
        opponent_elo = elo_ratings[opponent_id]
    
        # Расчет ожидаемого результата (Expected Score)
        expected_score_team = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))
        expected_score_opponent = 1 - expected_score_team

                        
        game_stats.loc[index, 'ELO_old'] = elo_ratings[team_id]
        game_stats.loc[index, 'ELO_O_old'] = elo_ratings[opponent_id]
            
            # Обновление ELO рейтинга на основе реального результата
        if result == 'W':
            elo_ratings[team_id] = team_elo + K_FACTOR * (1 - expected_score_team)
            elo_ratings[opponent_id] = opponent_elo + K_FACTOR * (0 - expected_score_opponent)
        elif result == 'L':
            elo_ratings[team_id] = team_elo + K_FACTOR * (0 - expected_score_team)
            elo_ratings[opponent_id] = opponent_elo + K_FACTOR * (1 - expected_score_opponent)
        else:
            elo_ratings[team_id] = team_elo + K_FACTOR * (0.5 - expected_score_team)
            elo_ratings[opponent_id] = opponent_elo + K_FACTOR * (0.5 - expected_score_opponent)
    
        # Присваиваем рейтинг командам в таблице
        game_stats.loc[index, 'ELO'] = elo_ratings[team_id]
        game_stats.loc[index, 'ELO_O'] = elo_ratings[opponent_id]
            
        game_stats.loc[index,'E_S'] = expected_score_team
        game_stats.loc[index,'E_S_O'] = expected_score_opponent

    #game_stats.to_csv('game_stats_one_r.csv', index=False, float_format='%.2f')
    
    return game_stats

#Функция для вычисления среднеквадратичной ошибки (MSE)
def calculate_mse(expected_results_team, expected_results_opponent, actual_results_team, actual_results_opponent):
    expected_results_team = np.array(expected_results_team)
    expected_results_opponent = np.array(expected_results_opponent)
    actual_results_team = np.array(actual_results_team)
    actual_results_opponent = np.array(actual_results_opponent)
    
    # Вычисление среднеквадратичной ошибки
    mse = np.mean((expected_results_team - actual_results_team) ** 2 + (expected_results_opponent - actual_results_opponent) ** 2)
    
    return mse
#Функция для нахождения оптимального значения K-фактора.
def find_optimal_k_factor(game_stats, k_range=np.arange(30, 60, 1)):
    optimal_k = None
    min_mse = float('inf')
    mse_values = []

    for k in k_range:
        game_stats = calculate_elo('data/targeted/game_stats_one_r.csv', k)

        actual_results_team = []
        expected_results_team = []
        actual_results_opponent = []
        expected_results_opponent = []

        for index, match in game_stats.iterrows():
            result = match['result']
            expected_result_team = match['E_S']
            expected_result_opponent = match['E_S_O']

            expected_results_team.append(expected_result_team if result == 'W' else (0.5 if result == 'D' else 0))
            expected_results_opponent.append(expected_result_opponent if result == 'L' else (0.5 if result == 'D' else 0))

            actual_results_team.append(1 if result == 'W' else (0.5 if result == 'D' else 0))
            actual_results_opponent.append(0 if result == 'W' else (0.5 if result == 'D' else 1))

        mse = calculate_mse(expected_results_team, expected_results_opponent, actual_results_team, actual_results_opponent)
        mse_values.append(mse)
        if mse < min_mse:
            min_mse = mse
            optimal_k = k

    print(f"Оптимальный K-фактор найден: {optimal_k}")
    return optimal_k
