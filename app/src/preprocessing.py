import pandas as pd
import warnings
import numpy as np
import os
import chardet

warnings.filterwarnings('ignore')

def replace_data(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep=";")
            df.replace(0, np.nan, inplace=True)
            df.to_csv(file_path, index=False, sep=";")

def re_format_file(folder_path):
    # Проход по всем файлам в папке
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):  # Проверяем, что файл CSV
            file_path = os.path.join(folder_path, filename)
            
            # Определяем кодировку
            with open(file_path, "rb") as f:
                rawdata = f.read()
                result = chardet.detect(rawdata)
                encoding = result["encoding"]

            print(f"Файл: {filename} | Определённая кодировка: {encoding}")

            # Читаем файл с определённой кодировкой и сохраняем в UTF-8
            if encoding:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    print(f"✅ {filename} перекодирован в UTF-8")
                except Exception as e:
                    print(f"❌ Ошибка при обработке {filename}: {e}")
            else:
                print(f"❌ Кодировка {filename} не определена, пропускаем файл.")

#Удаляем те игры, в которых ID team = !2
def remove_invalid_games(df):

    # группируем данные по номеру игры
    grouped_data = df.groupby('ID game')

    # список для хранения номеров игр, которые нужно удалить
    games_to_delete = []

    # Проверяем кол-во уникальных команд для каждой игры
    for name, group in grouped_data:
        if group['ID team'].nunique() != 2:
            games_to_delete.append(name)

    # Удаляем игры
    cleaned_data = df[~df['ID game'].isin(games_to_delete)]
    
    # Подсчет кол-ва удаленных игр
    removed_games_count = len(games_to_delete)
        
    # Подсчет общего кол-ва игр после удаления
    total_games_after = len(cleaned_data['ID game'].unique())
    print("Результат работы функции remove_invalid_games:")
    print(f"Количество удаленных игр: {removed_games_count}")
    print(f"Количество оставшихся игр: {total_games_after}", "\n")
    
    return cleaned_data

#Удаление игр с дифицитом данных
def process_game_data(df_player_game):
    # Группируем данные по номеру игры
    grouped_data = df_player_game.groupby('ID game')

    # Списки для хранения номеров игр
    games_to_delete = set()
    games_checked_first_condition = set()

    # Первая проверка: оставляем только игры, где у обеих команд есть хотя бы 4 игрока с `total time on ice`
    for name, group in grouped_data:
        teams_with_time = {
            team_id
            for team_id in group['ID team'].unique()
            if group[group['ID team'] == team_id]['total time on ice'].notnull().sum() >= 4
        }
        if len(teams_with_time) == 2:
            games_checked_first_condition.add(name)
        else:
            games_to_delete.add(name)

    # Вторая проверка: оставляем только игры, где у каждой команды есть минимум 3 игрока с любыми показателями
    for name in games_checked_first_condition:
        group = grouped_data.get_group(name)
        if group['ID team'].nunique() != 2:
            games_to_delete.add(name)
            continue

        teams_with_info = {
            team_id
            for team_id in group['ID team'].unique()
            if (group[group['ID team'] == team_id][['pucks', 'a shot on target', 'throws by', 'total time on ice']]
                .notnull().sum(axis=1) > 0).sum() >= 3
        }
        if len(teams_with_info) < 2:
            games_to_delete.add(name)

    # Удаляем выбранные игры
    cleaned_data = df_player_game[~df_player_game['ID game'].isin(games_to_delete)]
    
    # Вычисляем статистику
    removed_games_count = len(games_to_delete)
    total_games_after = cleaned_data['ID game'].nunique()

    # Вывод информации
    print("Результат работы функции process_game_data:")
    print(f"Количество удаленных игр: {removed_games_count}")
    print(f"Общее количество игр после удаления: {total_games_after}", "\n")

    return cleaned_data

# Удаление лишних игркоков, у которых нет амплуа
def process_player_amplua(df_player_amplua, games_data):
       
    # уникальные игроки
    unique_players = games_data['ID player'].unique()
    
    # Фильтрация данных
    filtered_players_data = df_player_amplua[df_player_amplua['ID player'].isin(unique_players)]
    
    # Добавление амплуа к статистике
    merged_data = pd.merge(games_data, filtered_players_data, on='ID player', how='left')
    
    return merged_data

#Проверка есть ли те игроки, которые были когдато не вратарями и создание compile_stats
#Те игроки, которые раньше были не вратарями, теперь имеют амплуа = 10 или 9
def check_and_modify_amplua(merged_data):

    # Выбор игроков с амплуа 8
    players_amp8 = merged_data.loc[merged_data['amplua'] == 8]

    # Если хотя бы одно значение в столбце 'total time on ice' > 1
    nonzero_columns = players_amp8[['total time on ice']].gt(1).any(axis=1)

    # Подсчет количества игроков, удовлетворяющих условию
    num_not_empty_values = nonzero_columns.sum()
    print("Результат работы функции check_and_modify_amplua:")
    print("Количество игроков, удовлетворяющих условию:", num_not_empty_values, "\n")

    # Выбор игроков, которым нужно изменить амплуа
    players_to_change = players_amp8[nonzero_columns]

    # Случайное присвоение амплуа 9 или 10
    merged_data.loc[players_to_change.index, 'amplua'] = np.random.choice([9, 10], size=len(players_to_change))

    # Удаление столбца 'pucks'
    merged_data.drop('pucks', axis=1, inplace=True)
    
    return merged_data

def remove_inconsistent_games(df):
    inconsistent_games = set()

    for (game_id, team_id), group in df.groupby(['ID game', 'ID team']):
        # Формируем множества, исключая нулевые значения в IDp_in_ice
        # Исключаем нулевые и пустые значения
        unique_in_ice = {x for x in group['IDp_in_ice'] if pd.notna(x) and x != 0}
        unique_out_ice = {x for x in group['IDp_out_ice'] if pd.notna(x) and x != 0}


        # Если вратари в этих столбцах разные и их больше одного, игра считается некорректной
        if len(unique_in_ice) > 1 or len(unique_out_ice) > 1 and unique_in_ice != unique_out_ice:
            inconsistent_games.add(game_id)

    # Отбираем удаленные строки для анализа
    removed_rows = df[df['ID game'].isin(inconsistent_games)]

    # Фильтруем DataFrame, удаляя некорректные игры
    cleaned_df = df[~df['ID game'].isin(inconsistent_games)]

    print("Результат работы фукнции remove_inconsistent_games:")
    print(f"Удалено игр:{len(removed_rows)}", "\n")

    #print(f"Удалены игры: {sorted(inconsistent_games)}")
    #print("Первые 10 удаленных строк:")
    #print(removed_rows.head(10))

    cleaned_df.to_csv("data/interim/remove_inconsistent_games.csv", index = False)

    return cleaned_df

#Удаление тех ID game из таблицы goalkeeper_event, которых нет в таблице compile_stats
def filter_goalkeeper_event_by_compile_stats(goalkeeper_event, compile_stats):
    # Получаем уникальные ID игр из обоих столбцов
    common_games = set(goalkeeper_event['ID game']) & set(compile_stats['ID game'])
    
    # Фильтруем таблицу goalkeeper_event, оставляя только игры, которые есть в обоих
    goalkeeper_event_filtered = goalkeeper_event[goalkeeper_event['ID game'].isin(common_games)]
    compl_filtred = compile_stats[compile_stats['ID game'].isin(common_games)]

    print("Результат работы функции filter_goalkeeper_event_by_compile_stats:")
    print(f"Количество общих игр в обеих таблицах: {len(common_games)}", "\n")
    
    goalkeeper_event_filtered.to_csv("data/interim/goalkeeper_event_filtered.csv", index=False)
    compl_filtred.to_csv("data/interim/compl_filtred.csv", index=False)

    return goalkeeper_event_filtered, compl_filtred

def merge_goalkeeper_events(compile_stats, goalkeeper_event):
    new_rows = []  # Список для добавления новых вратарей
    games_to_remove = []  # Игры, где обоих вратарей нет в goalkeeper_event

    # Проход по каждой команде в каждой игре
    for (game_id, team_id), group in goalkeeper_event.groupby(['ID game', 'ID team']):
        # Вратари из таблицы goalkeeper_event (те, кто реально выходил на лед)
        if (group['IDp_in_ice'] != 0).any():
            goalkeepers_in_event = set(group['IDp_in_ice']) - {0}
        elif (group['IDp_out_ice'] != 0).any():
            goalkeepers_in_event = set(group['IDp_out_ice']) - {0}
        else:
            goalkeepers_in_event = set()

        # Вратари из compile_stats (проверяем по ID игрока, а не по амплуа)
        players_in_stats = compile_stats.loc[
            (compile_stats['ID game'] == game_id) & (compile_stats['ID team'] == team_id),
            ['ID player', 'amplua']
        ]
        
        goalkeepers_in_stats = set(players_in_stats.loc[players_in_stats['amplua'] == 8, 'ID player'])
        
        # Удаляем дублирующиеся строки, если игрок уже есть в команде в этом матче
        compile_stats = compile_stats.drop(
            compile_stats[
                (compile_stats['ID game'] == game_id) & 
                (compile_stats['ID team'] == team_id) & 
                (compile_stats['ID player'].isin(goalkeepers_in_event)) &
                (compile_stats['amplua'] != 8)
            ].index
        )

        # Теперь обновляем амплуа для оставшихся игроков
        compile_stats.loc[
            (compile_stats['ID game'] == game_id) & 
            (compile_stats['ID team'] == team_id) & 
            (compile_stats['total time on ice'] == 0) &
            (compile_stats['throws by'] == 0) &
            (compile_stats['a shot on target'] == 0) &
            (compile_stats['blocked throws'] == 0) &
            (compile_stats['ID player'].isin(goalkeepers_in_event)),
            'amplua'
        ] = 8

        # Если у команды два вратаря в compile_stats, но один из них не в goalkeeper_event
        if len(goalkeepers_in_stats) > 1:
            keepers_to_remove = goalkeepers_in_stats - goalkeepers_in_event  # Кого нет в event
            keepers_to_keep = goalkeepers_in_stats & goalkeepers_in_event  # Кого можно оставить

            if keepers_to_keep:
                compile_stats = compile_stats[~(
                    (compile_stats['ID game'] == game_id) & 
                    (compile_stats['ID team'] == team_id) & 
                    (compile_stats['ID player'].isin(keepers_to_remove)) &
                    (compile_stats['amplua'] == 8)  # Только вратарей!
                )]
            else:
                games_to_remove.append(game_id)
                continue  # Переход к следующей игре

        # Если вратаря в compile_stats нет, но он есть в goalkeeper_event — добавляем
        if not goalkeepers_in_stats and goalkeepers_in_event:
            for gk in goalkeepers_in_event:
                new_rows.append({'ID game': game_id, 'ID team': team_id, 'ID player': gk, 'amplua': 8})

    # Удаляем игры, где обоих вратарей нет в goalkeeper_event
    compile_stats = compile_stats[~compile_stats['ID game'].isin(games_to_remove)]

    # Добавляем новых вратарей
    if new_rows:
        compile_stats = pd.concat([compile_stats, pd.DataFrame(new_rows)], ignore_index=True)

    print("Результат работы функции merge_goalkeeper_events:")
    print(f"Удалено игр с обоими неправильными вратарями: {len(games_to_remove)}")
    print(f"Добавлено вратарей в игры: {len(new_rows)}")

    compile_stats.to_csv("data/interim/merge_goalkeeper_events.csv", index=False)

    return compile_stats


#Добавим возраст к игрокам
def add_age_to_players_stats(compile_stats, player_age):

    # Получение списка уникальных игроков из таблицы с игроками
    unique_players = compile_stats['ID player'].unique()

    # Фильтрация данных
    filtered_players_data = player_age[player_age['ID player'].isin(unique_players)]

    # Добавление возраста к статистике игроков
    merged_data = pd.merge(compile_stats, filtered_players_data, on='ID player', how='left')
    
    return merged_data

#Удаляем тех у кого путстые строки кроме вратарей
def remove_empty_rows(merged_data):
    # Создаем условия для фильтрации строк
    condition1 = (merged_data['amplua'] != 8)
    condition2 = (merged_data['total time on ice'] < 10) & \
                 (merged_data['throws by'].isnull()) & \
                 (merged_data['a shot on target'].isnull()) & \
                 (merged_data['blocked throws'].isnull())

    # строк до удаления
    num_rows_before_removal = len(merged_data)

    # фильтруем, удаляем
    merged_data = merged_data[~(condition1 & condition2)]

    # кол-во удаленных строк
    num_rows_deleted = num_rows_before_removal - len(merged_data)
    print("Результат работы функции remove_empty_rows:")
    print(f"Количество удаленных строк: {num_rows_deleted}", "\n")

    merged_data.to_csv('data/targeted/compile_stats.csv', index=False)

    return merged_data

#Подготовка таблицы plus_minus для мержа к статиcтике
#Удаление событий, с дифицитом данных
#Валидация данных, таблицы +/-, проверка на пустые ID team, на кол-во игрков в одном событии, а также, чтобы было больше одной
#команды в одном событии
def validation_for_pm(game_plus_minus):
    # Удаляем строки, в которых ID team равен 0
    game_plus_minus_cleaned = game_plus_minus[game_plus_minus['ID team'] != 0]
    
    # Сгруппируем данные по столбцам 'ID event' и 'ID team' и посчитаем колво строк в каждой группе
    event_team_row_counts = game_plus_minus_cleaned.groupby(['ID event', 'ID team']).size().reset_index(name='row_count')
    
    # Проверяем, если в одном событии количество разных команд не равно 2, то удаляем это событи
    event_teams_counts = event_team_row_counts.groupby('ID event').size()
    incomplete_event_ids = event_teams_counts[event_teams_counts != 2].index
    game_plus_minus_cleaned = game_plus_minus_cleaned[~game_plus_minus_cleaned['ID event'].isin(incomplete_event_ids)]
    
    # Проверяем, чтобы в одном событии в каждй команде было не менее трех игроков, если их меньше, то удаляем это событие
    event_player_counts = game_plus_minus_cleaned.groupby(['ID event', 'ID team'])['ID player'].nunique().reset_index(name='player_count')
    incomplete_event_ids = event_player_counts[event_player_counts['player_count'] < 3]['ID event'].unique()
    game_plus_minus_cleaned = game_plus_minus_cleaned[~game_plus_minus_cleaned['ID event'].isin(incomplete_event_ids)]
    
    # Выводим информацию
    removed_rows_count = len(game_plus_minus) - len(game_plus_minus_cleaned)
    print("Результат работы функции validation_for_pm:")
    print(f"Количество удаленных строк: {removed_rows_count}")
    print(f"Размерность таблицы после удаления неполных событий: {game_plus_minus_cleaned.shape}", "\n")
    
    # Считаем кол-во удаленных ID event и кол-во игр с неполными данными
    incomplete_events_after = len(incomplete_event_ids)
    incomplete_games = len(game_plus_minus_cleaned['ID game'].unique())
    
    return game_plus_minus_cleaned


#Удаление тех ID game которых для которых нету +/- вообще
def remove_PL_game(compile_stats, plus_minus_player_game):

    # Получение списка уникальных ID game из файла plus_minus_player_game
    valid_game_ids = plus_minus_player_game['ID game'].unique()

    # Фильтрация данных в compile_stats по ID game, оставляем только те строки, которые присутствуют в plus_minus_player_game
    compile_stats_filtered = compile_stats[compile_stats['ID game'].isin(valid_game_ids)]

    return compile_stats_filtered


#Сумма, подсчет +/- для игрокв в каждом матче
def calculate_plus_minus(df_plus_minus):

    # Создаем пустой датафрейм для хранения результатов
    df_pm = pd.DataFrame(columns=['ID season', 'ID game', 'ID team', 'ID player', 'p/m'])

    # пустой словарь для хранения информации о том, какие игроки уже были учтены в каждой игре
    players_in_game = {}

    # пустой словарь для хранения информации о командах каждого игрока и номере сезона
    player_teams = {}

    # проходим по строкам в df_plus_minus
    for idx, row in df_plus_minus.iterrows():
        game_id = row['ID game']
        team_id = row['ID team']
        player_id = row['ID player']
        season_id = row['ID season']
        scoring_team = row['the scoring team']
        pm = 1 if team_id == scoring_team else -1  # Плюс или минус в зависимости от команды, забившей гол

        # Проверяем, был ли игрок уже учтен в этой игре
        if (game_id, player_id) not in players_in_game:
            # Если игрока нет, добавляем его в словарь результатов
            players_in_game[(game_id, player_id)] = pm

        # Проверяем, есть ли уже информация о команде и сезоне для этого игрока
        if (game_id, player_id) not in player_teams:
            player_teams[(game_id, player_id)] = {'team_id': team_id, 'season_id': season_id, 'game_id': game_id}

        else:
            # Если игрок уже есть, обновляем его плюс-минус
            players_in_game[(game_id, player_id)] += pm

    # список для каждого игрока
    dfs = []
    for (game_id, player_id), pm in players_in_game.items():
        dfs.append(pd.DataFrame([{'ID season': player_teams[(game_id, player_id)]['season_id'],
                                  'ID game': game_id,
                                  'ID team': player_teams[(game_id, player_id)]['team_id'],
                                  'ID player': player_id,
                                  'p/m': pm}]))

    # Объединяем все датафреймы
    df_pm = pd.concat(dfs, ignore_index=True)

    return df_pm

#Добавляем показатель +/-
def add_plus_minus(merged_data, df_plus_minus):

    merged_data = pd.merge(merged_data, df_plus_minus, on=['ID game', 'ID team', 'ID player'], how='left')
    
    merged_data.drop('ID season', axis=1, inplace=True)

    return merged_data

# Проверка анамалий, если игрок по времени не был на льду а p/m есть, то удаляем эту игру
def remove_games_with_missing_time(df):
    
    # условия для удаления
    condition = (df['p/m'].notnull()) & (df['total time on ice'].isnull())

    # спис ID game, которые будут удалены
    games_deleted = df.loc[condition, 'ID game'].unique()

    # Удаление строк, удовлетворяющих условию
    cleaned_df = df.loc[~condition].copy()

    ##############################
    # Подсчет кол-ва удаленных игр
    removed_games_count = len(games_deleted)
    
    # Подсчет общего кол-ва игр после удаления
    total_games_after = len(cleaned_df['ID game'].unique())
    print("Результат работы функции remove_games_with_missing_time:")
    print(f"Количество удаленных игр: {removed_games_count}")
    print(f"Общее количество игр после удаления: {total_games_after}", "\n")

    return cleaned_df

# Дабавление голов и пассов
def add_goals_and_assists(df_merged, df_goals_and_passes):

    # Груп таблицы goals_and_passes по 'ID game', 'ID team', 'ID player scored' для подсчета голов
    goals_grouped = df_goals_and_passes.groupby(['ID game', 'ID team', 'ID player scored']).size().reset_index(name='goals')

    # Груп таблицы goals_and_passes по 'ID game', 'ID team', 'ID player assist' для подсчета ассистов
    assists_grouped = df_goals_and_passes.groupby(['ID game', 'ID team', 'ID player assist']).size().reset_index(name='assists')

    # Объединение результатов подсчета голов и голевых передач с таблицей df_merged
    merged_data = pd.merge(df_merged, goals_grouped, left_on=['ID game', 'ID team', 'ID player'], right_on=['ID game', 'ID team', 'ID player scored'], how='left')
    merged_data = pd.merge(merged_data, assists_grouped, left_on=['ID game', 'ID team', 'ID player'], right_on=['ID game', 'ID team', 'ID player assist'], how='left')

    # Заполнение пропущенных значений нулями
    merged_data['goals'] = merged_data['goals'].fillna(0)
    merged_data['assists'] = merged_data['assists'].fillna(0)

    # Удаление лишних столбцов, если они были добавлены из goals_grouped и assists_grouped
    merged_data.drop(['ID player scored', 'ID player assist'], axis=1, inplace=True, errors='ignore')

    return merged_data

#те игры в которы больше двух вратарей зарегано
#те игры , в которых была замена
def filter_goalkeeper_events(compile_stats, goalkeeper_event, output_filtered_path, output_replacements_path):
    
    # Группировка по ID game и амплуа, подсчет количества игроков
    player_counts = compile_stats.groupby(['ID game', 'amplua']).size().unstack(fill_value=0)
    
    # Подсчет количества игроков с амплуа 8 в каждой игре
    amplua_8_counts = player_counts[8]
    
    # Выбор игр, где количество игроков с амплуа 8 больше двух
    selected_games = amplua_8_counts[amplua_8_counts != 2]
    
    # Получение ID игр
    selected_game_ids = selected_games.index.tolist()
    
    # Фильтрация данных по выбранным играм
    selected_compile_stats = compile_stats[compile_stats['ID game'].isin(selected_game_ids)]
    
    # Получение списка уникальных ID game из таблицы compile_stats
    unique_game_ids = selected_compile_stats['ID game'].unique()
    
    # Фильтрация goalkeeper_event по уникальным ID game из compile_stats
    filtered_goalkeeper_event = goalkeeper_event[goalkeeper_event['ID game'].isin(unique_game_ids)]
    
    # Сохранение отфильтрованных данных в CSV файл
    filtered_goalkeeper_event.to_csv(output_filtered_path, index=False)
    
    # Подсчет и вывод количества уникальных ID game
    unique_game_count = filtered_goalkeeper_event['ID game'].nunique()
    print("Результат работы функции filter_goalkeeper_events:")
    print("Количество уникальных ID game (filtered):", unique_game_count)
    
    # Группировка данных
    grouped_by_game_team = filtered_goalkeeper_event.groupby(['ID game', 'ID team'])
    
    # Список для хранения ID game, где была замена вратарей
    games_with_goalkeeper_change = []
    
    # Перебор каждой группы
    for (game_id, team_id), group in grouped_by_game_team:
        # Если в группе больше одной записи, значит была замена вратарей
        if len(group) > 1:
            # Проверяем, разные ли значения (кроме 0) в колонках IDp_out_ice и IDp_in_ice
            unique_values_out = group['IDp_out_ice'][group['IDp_out_ice'] != 0].nunique()
            unique_values_in = group['IDp_in_ice'][group['IDp_in_ice'] != 0].nunique()
            if (unique_values_out > 0 and unique_values_in > 0) and (unique_values_out != 1 or unique_values_in != 1):
                games_with_goalkeeper_change.append(game_id)
    
    # Создание DataFrame для сохранения игр с заменой вратарей
    goalkeeper_change_df = filtered_goalkeeper_event[filtered_goalkeeper_event['ID game'].isin(games_with_goalkeeper_change)]
    
    # Сохранение данных о заменах вратарей в CSV файл
    goalkeeper_change_df.to_csv(output_replacements_path, index=False)
    
    # Подсчет и вывод количества уникальных ID game с заменой вратарей
    unique_game_count_with_changes = goalkeeper_change_df['ID game'].nunique()
    print("Количество уникальных ID game (с заменой вратарей):", unique_game_count_with_changes, "\n")


def del_swap_goalk(filtered_event_path, goalkeeper_replace_path, compile_stats, output_filtered_event_path, output_compile_stats_path):
    # Чтение данных из файлов
    filtered_goalkeeper_event = pd.read_csv(filtered_event_path)
    goalkeeper_replace = pd.read_csv(goalkeeper_replace_path)
    
    # Удаление игр из filtered_goalkeeper_event, которые есть в goalkeeper_replace по ID game
    filtered_goalkeeper_event = filtered_goalkeeper_event[~filtered_goalkeeper_event['ID game'].isin(goalkeeper_replace['ID game'])]
    
    # Сохранение отфильтрованных данных в CSV файл
    filtered_goalkeeper_event.to_csv(output_filtered_event_path, index=False)
    
    # Удаление тех вратарей, которые не участвовали в игре
    # Загрузка данных из таблиц
    filtered_goalkeeper_event_filtered = pd.read_csv(output_filtered_event_path)
    compile_stats_filtered = compile_stats
    
    rows_to_remove = []
    
    # Перебор каждой строки в filtered_goalkeeper_event_filtered
    for index, row in filtered_goalkeeper_event_filtered.iterrows():
        game_id = row['ID game']
        team_id = row['ID team']
        player_id = row['IDp_in_ice']
        
        # Поиск всех игроков с амплуа = 8 в данной игре и команде
        players_amplua_8 = compile_stats_filtered[(compile_stats_filtered['ID game'] == game_id) & 
                                                  (compile_stats_filtered['ID team'] == team_id) & 
                                                  (compile_stats_filtered['amplua'] == 8)]
        
        # Перебор каждого игрока с амплуа = 8 в данной игре и команде
        for _, player in players_amplua_8.iterrows():
            # Проверка, не является ли текущий игрок тем, кто участвовал в игре
            if player_id != player['ID player'] and player_id != 0:
                # Добавление индекса строки для удаления в список
                rows_to_remove.append(player.name)
    
    # Удаление строк из compile_stats_filtered по индексам
    compile_stats_filtered.drop(rows_to_remove, inplace=True)
    
    # Сохранение отфильтрованных данных в CSV файл
    compile_stats_filtered.to_csv(output_compile_stats_path, index=False)
    
    # Удаление строк из compile_stats_filtered, где ID game присутствует в списке games_to_remove
    games_to_remove = goalkeeper_replace['ID game'].unique()
    
    # Удаление строк из compile_stats, где ID game присутствует в списке games_to_remove
    compile_stats_filtered = compile_stats[~compile_stats['ID game'].isin(games_to_remove)]
    
    # Сохранение отфильтрованных данных в CSV файл
    compile_stats_filtered.to_csv(output_compile_stats_path, index=False)

def clean_compile_stats(file_path):
    # Чтение данных из файла
    compile_stats = pd.read_csv(file_path)
    
    # Удаление игр, в которых количество игроков с amplua = 8 больше двух
    # Подсчет количества игроков с amplua = 8 для каждой игры
    amplua_8_counts = compile_stats[compile_stats['amplua'] == 8].groupby('ID game').size()
    # Получение списка ID игр, где количество игроков с amplua = 8 больше двух
    games_to_remove = amplua_8_counts[amplua_8_counts > 2].index.tolist()
    # Удаление игр
    compile_stats = compile_stats[~compile_stats['ID game'].isin(games_to_remove)]
    
    unique_game_count = compile_stats['ID game'].nunique()

    print("Результат работы функции clean_compile_stats:")
    print(f"ИГры с двумя вратарями:{games_to_remove}")
    print(f"списка ID игр, где количество игроков с amplua = 8 больше двух:{len(games_to_remove)}")
    print("Количество уникальных ID game после удаления игр с более чем двумя игроками amplua = 8:", unique_game_count)
    
    # # Замена пустых значений на 0
    # compile_stats.fillna(0, inplace=True)
    
    # # Удаление строк, в которых игроки имеют амплуа, отличное от 8, и все указанные столбцы содержат нулевые значения
    # compile_stats = compile_stats[~((compile_stats['amplua'] != 8) & (
    #         compile_stats[['total time on ice', 'throws by', 'a shot on target', 'blocked throws', 'p/m', 'goals', 'assists']].eq(0).all(axis=1)
    # ))]
    
    # Сохранение отфильтрованных данных в CSV файл
    compile_stats.to_csv(file_path, index=False)
    
    unique_game_count = compile_stats['ID game'].nunique()
    print("Количество уникальных ID game после финальной фильтрации:", unique_game_count, "\n")

#Проверка на странные игры, нужно доделать очистку
def clean_problem(file_path, output_path):
    # Чтение исходного файла
    compile_stats = pd.read_csv(file_path)

    # Список столбцов с игровыми показателями для проверки
    columns_to_check = ['total time on ice', 'throws by', 'a shot on target', 'blocked throws', 'p/m', 'goals', 'assists']
    
    # Заменяем нули на NaN (чтобы их можно было обнаружить как отсутствующие значения)
    compile_stats[columns_to_check] = compile_stats[columns_to_check].replace({0: None})

    # ===============================================
    # 1. Исправление записей: если у игрока хотя бы один из показателей равен 0 или отсутствует,
    #    и в его команде (в данной игре) нет игрока с амплуа = 8, то назначаем ему амплуа = 8.
    # ===============================================
    # Перебираем по каждой игре
    for game_id in compile_stats['ID game'].unique():
        game_data = compile_stats[compile_stats['ID game'] == game_id]
        # Перебираем по каждой команде в игре
        for team_id in game_data['ID team'].unique():
            team_data = game_data[game_data['ID team'] == team_id]
            # Если в команде еще нет игрока с амплуа 8, то ищем кандидата для замены
            if not any(team_data['amplua'] == 8):
                # Находим индексы игроков, у которых хотя бы один из указанных показателей отсутствует
                indices_to_fix = team_data[team_data[columns_to_check].isna().any(axis=1)].index
                if len(indices_to_fix) > 0:
                    # Присваиваем амплуа 8 этим игрокам
                    compile_stats.loc[indices_to_fix, 'amplua'] = 8

    # ===============================================
    # 2. Теперь выполняем проверку и удаление "странных" игр:
    #    - Игры, в которых общее количество игроков с амплуа = 8 меньше двух.
    #    - Игры, в которых хотя бы в одной команде количество игроков с амплуа = 8 не равно 1.
    # ===============================================

    # Пересчёт количества игроков с амплуа 8 по играм
    players_with_amplua_8 = compile_stats[compile_stats['amplua'] == 8]\
                               .groupby('ID game')\
                               .size()\
                               .reset_index(name='count_amplua_8')
    games_with_few_players = players_with_amplua_8[players_with_amplua_8['count_amplua_8'] < 2]
    game_ids_with_few_players = games_with_few_players['ID game'].tolist()
    print("ID игр с количеством игроков с амплуа равным 8 меньше двух:", game_ids_with_few_players)

    # Пересчёт количества игроков с амплуа 8 по каждой команде в игре
    players_with_amplua_8_team = compile_stats[compile_stats['amplua'] == 8]\
                                    .groupby(['ID game', 'ID team'])\
                                    .size()\
                                    .reset_index(name='count_amplua_8')
    games_with_inconsistent_amplua_count = players_with_amplua_8_team[players_with_amplua_8_team['count_amplua_8'] != 1]['ID game'].unique()
    print("ID игр, в которых у какой-то команды количество игроков с амплуа 8 не равно 1:", games_with_inconsistent_amplua_count)

    # Объединяем ID игр, подлежащих удалению
    all_games_to_remove = set(game_ids_with_few_players).union(set(games_with_inconsistent_amplua_count))
    compile_stats = compile_stats[~compile_stats['ID game'].isin(all_games_to_remove)]

    # Удаляем также игры с заранее заданными ID
    specific_game_ids_to_remove = [7273, 10138, 9484, 1111, 8914]
    compile_stats = compile_stats[~compile_stats['ID game'].isin(specific_game_ids_to_remove)]

    # Сохранение итоговых данных в указанный файл
    compile_stats.to_csv(output_path, index=False)

    unique_game_count = compile_stats['ID game'].nunique()
    print("Количество уникальных ID game после очистки:", unique_game_count, "\n")

#Формирование таблицы goalkeeper_stats , на основе нашей таблици compile_stats
#Создание таблицы goalkeepers и подсчет статитики для вратарей
def create_goalkeepers_table(compile_stats_path, goals_and_passes_path, output_path):
    # Чтение данных из файла compile_stats
    compile_stats = pd.read_csv(compile_stats_path)
    
    # Фильтруем данные по условию amplua == 8 и копируем
    goalkeepers = compile_stats[compile_stats['amplua'] == 8][['ID game', 'ID team', 'ID player']].copy()
    
    goalkeepers['missed pucks'] = None
    goalkeepers['total throws'] = None
    
    # Создаем таблицу goalkeepers_data
    goalkeepers_data = pd.DataFrame(columns=['ID game', 'ID team', 'ID player', 'missed pucks', 'total throws'])
    
    # Перебираем строки compile_stats, чтобы посчитать total throws и missed pucks
    for index, row in goalkeepers.iterrows():
        game_id = row['ID game']
        player_id = row['ID player']
        team_id = row['ID team']
        
        # Находим количество бросков против команды в игре
        opponent_total_throws = compile_stats[(compile_stats['ID game'] == game_id) & (compile_stats['ID team'] != team_id)]['a shot on target'].sum()
        
        # Добавляем данные в goalkeepers_data
        goalkeepers_data.loc[len(goalkeepers_data)] = [game_id, team_id, player_id, None, opponent_total_throws]
    
    # Чтение данных из файла goals_and_passes
    goals_and_passes = pd.read_csv(goals_and_passes_path, sep=";")
    
    # Перебираем строки goals_and_passes, чтобы посчитать missed pucks
    for index, row in goalkeepers_data.iterrows():
        game_id = row['ID game']
        team_id = row['ID team']
        
        # Считаем missed pucks для каждой игры и команды
        missed_pucks = len(goals_and_passes[(goals_and_passes['ID game'] == game_id) & (goals_and_passes['ID team'] != team_id)])
        
        # Обновляем данные
        goalkeepers_data.loc[(goalkeepers_data['ID game'] == game_id) & (goalkeepers_data['ID team'] == team_id), 'missed pucks'] = missed_pucks
    
    # Рассчитываем процент отраженных бросков
    goalkeepers_data['% of reflected shots'] = (1 - (goalkeepers_data['missed pucks'] / goalkeepers_data['total throws'])) * 100
    
    goalkeepers_data.to_csv(output_path, index=False, float_format='%.2f')