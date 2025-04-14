import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append(r'app/src')

#from app.src.method_bayesian import predict_match_outcome

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

def invert_result(result):
    """
    Если результат записан для первой команды, то для второй команды он инвертируется:
      - Если 'W' (победа) для первой, то для второй будет 'L'
      - Если 'L' (поражение) для первой, то для второй будет 'W'
      - Если 'D' (ничья), то остаётся 'D'
    """
    if result == 'W':
        return 'L'
    elif result == 'L':
        return 'W'
    else:
        return result  # предполагаем, что ничья обозначается 'D'

def get_team_results(df, team_id):
    """
    Функция возвращает серию результатов для команды team_id и количество игр,
    в которых она принимала участие.
    
    Рассматриваются игры, где команда выступала как первая (ID team) и как вторая (ID opponent).
    Для игр, где команда играет вторым, результат инвертируется.
    """
    # Игры, где команда указана как первая
    mask_first = df["ID team"] == team_id
    df_first = df[mask_first].copy()
    results_first = df_first["result"]
    
    # Игры, где команда указана как вторая
    mask_second = df["ID opponent"] == team_id
    df_second = df[mask_second].copy()
    # Инвертируем результат для игр, где команда играет второй
    results_second = df_second["result"].apply(invert_result)
    
    # Количество игр, где участвовала команда
    game_count = df[(df["ID team"] == team_id) | (df["ID opponent"] == team_id)].shape[0]
    
    # Объединяем результаты
    results = pd.concat([results_first, results_second])
    
    return results, game_count

def calc_win_probability(results):
    """
    Вычисляет вероятность победы по результатам игр.
    Вероятность победы = число побед / общее число игр.
    Если игр нет, возвращается 0.
    """
    total_games = len(results)
    if total_games == 0:
        return 0.0
    wins = (results == 'W').sum()
    return wins / total_games

def calc_recent_win_probability(df, team_id, recent_games=5):
    """
    Вычисляет вероятность победы команды за последние recent_games.
    
    Для этого:
      1. Фильтруем игры, где участвовала команда.
      2. Приводим столбец с датой к типу datetime (если ещё не приведён).
      3. Сортируем по дате, выбираем последние recent_games игр.
      4. Вычисляем вероятность победы по тем же правилам.
    """
    # Если столбец date в формате строки, приводим его к datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
    
    # Игры, где команда выступала как первая
    mask_first = df["ID team"] == team_id
    df_first = df[mask_first].copy()
    df_first["Result_local"] = df_first["result"]
    
    # Игры, где команда выступала как вторая
    mask_second = df["ID opponent"] == team_id
    df_second = df[mask_second].copy()
    df_second["Result_local"] = df_second["result"].apply(invert_result)
    
    # Объединяем обе группы игр
    df_team = pd.concat([df_first, df_second])
    # Сортируем по дате от самой новой к старой
    df_team = df_team.sort_values(by="date", ascending=False)
    # Выбираем последние recent_games игр
    recent_df = df_team.head(recent_games)
    
    if recent_df.empty:
        return 0.0
    wins = (recent_df["Result_local"] == 'W').sum()
    return wins / len(recent_df)

def get_recent_games_info(df, team_id, recent_games=5):
    """
    Возвращает DataFrame с информацией о последних recent_games играх, в которых участвовала команда.
    Выводятся следующие столбцы:
      - ID game
      - ID team (то, за кого выступала команда)
      - ID opponent
      - Result_local – результат с точки зрения данной команды
      - date
    """
    df_temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_temp["date"]):
        df_temp["date"] = pd.to_datetime(df_temp["date"], errors='coerce')
    
    # Игры, где команда выступала как первая
    mask_first = df_temp["ID team"] == team_id
    df_first = df_temp[mask_first].copy()
    df_first["Result_local"] = df_first["result"]
    
    # Игры, где команда выступала как вторая
    mask_second = df_temp["ID opponent"] == team_id
    df_second = df_temp[mask_second].copy()
    df_second["Result_local"] = df_second["result"].apply(invert_result)
    
    # Объединяем и сортируем по дате в порядке убывания
    df_team = pd.concat([df_first, df_second])
    df_team = df_team.sort_values(by="date", ascending=False)
    
    # Выбираем последние recent_games игр
    recent_games_df = df_team.head(recent_games)
    # Оставляем нужные столбцы
    return recent_games_df[["ID game", "ID team", "ID opponent", "Result_local", "date"]]

def predict_match_outcome(df, team_A_id, team_B_id, recent_games=5):
    """
    Основная функция, которая принимает датафрейм с историей игр, ID двух команд и
    коэффициент (N), определяющий число последних игр для анализа.
    
    Алгоритм:
      1. Вычисляем априорную вероятность победы на основе всех игр.
      2. Вычисляем вероятность победы по последним recent_games матчам.
      3. Перемножаем априорные вероятности и вероятности последних игр.
      4. Нормализуем полученные значения так, чтобы их сумма была равна 1.
      5. Дополнительно возвращаем DataFrame с информацией о последних recent_games играх каждой из команд.
    
    Функция возвращает словарь с рассчитанными вероятностями и дополнительной информацией.
    """

    compile_stats_path = r"data/targeted/compile_stats.csv"
    game_history_path = r"data/raw/game_history.csv"
    df_compile = pd.read_csv(compile_stats_path)
    df_history = pd.read_csv(game_history_path, sep=";")

    df_rating = calc_team_rating(df_compile, df_history, season_id=None, team_ids=[team_A_id, team_B_id])

    rating_a = df_rating[df_rating["ID team"] == team_A_id]["team_rating"].values[0]
    rating_b = df_rating[df_rating["ID team"] == team_B_id]["team_rating"].values[0]

    # Апариорные вероятности побед (на основе всех игр)
    results_A, games_count_A = get_team_results(df, team_A_id)
    results_B, games_count_B = get_team_results(df, team_B_id)
    
    p_A_overall = calc_win_probability(results_A)
    p_B_overall = calc_win_probability(results_B)
    
    # Вероятности побед в последних матчах
    p_A_recent = calc_recent_win_probability(df, team_A_id, recent_games)
    p_B_recent = calc_recent_win_probability(df, team_B_id, recent_games)
    
    # Итоговые "сырые" баллы: перемножение априорных вероятностей и вероятностей последних игр
    score_A = p_A_overall * p_A_recent
    score_B = p_B_overall * p_B_recent
    
    # Если сумма равна 0, предотвращаем деление на 0
    if (score_A + score_B) == 0:
        return {
            "team_A_win_prob": None,
            "team_B_win_prob": None,
            "message": "Недостаточно данных для расчёта вероятности.",
            "recent_games_info": {"team_A": None, "team_B": None}
        }
    
    # Нормализация
    norm_A = score_A / (score_A + score_B)
    norm_B = score_B / (score_A + score_B)
    
    # Получаем информацию о последних играх для команды A и команды B
    recent_info_A = get_recent_games_info(df, team_A_id, recent_games)
    recent_info_B = get_recent_games_info(df, team_B_id, recent_games)
        
    return {
        "team_A_games_count": games_count_A,
        "team_A_rating": rating_a,
        "team_A_win_prob": round(norm_A, 2),
        "team_B_games_count": games_count_B,
        "team_B_rating": rating_b,
        "team_B_win_prob": round(norm_B, 2),
        "overall_probs": {"team_A": round(p_A_overall, 2), "team_B": round(p_B_overall, 2)},
        "recent_probs": {"team_A": round(p_A_recent, 2), "team_B": round(p_B_recent, 2)},
        "recent_games_info": {
            "team_A": recent_info_A,
            "team_B": recent_info_B
        }
    }

@st.cache_data
def load_game_data():
    df_games = pd.read_csv(r'data/targeted/game_stats_one_r.csv')
    #Удаляем игры где была ничья
    df_games = df_games[df_games["result"] != "D"].copy()
    return df_games

def bayesian_analysis():
    # Определяем словарь для переименования столбцов
    mapping = {
        "ID game": "ID игры",
        "ID team": "ID команды",
        "ID opponent": "ID соперника",
        "Result_local": "Исход",
        "date": "Дата"
    }
    st.title("Байесовский метод прогнозирования исхода матча")

    # Загружаем данные (допустим, используем лишь историю игр)
    df_game = load_game_data()
    available_teams = sorted(df_game["ID team"].unique())

    # Поля ввода: идентификаторы команд выбираются из выпадающего списка
    team_A_id = st.selectbox("Выберите ID команды A", options=available_teams, index=0)
    team_B_id = st.selectbox("Выберите ID команды B", options=available_teams, index=1)

    recent_games = st.number_input("Количество последних игр для анализа", min_value=1, max_value=60, value=5)

    #Названия команд
    team_info = pd.read_csv(r"data/targeted/team_ratings_merge.csv")

    # Определяем названия команд по ID
    team_A_name_series = team_info.loc[team_info['ID team'] == int(team_A_id), 'TEAM_NAME']
    team_A_name = team_A_name_series.values[0] if not team_A_name_series.empty else "Неизвестная"
    team_B_name_series = team_info.loc[team_info['ID team'] == int(team_B_id), 'TEAM_NAME']
    team_B_name = team_B_name_series.values[0] if not team_B_name_series.empty else "Неизвестная"
    
    if st.button("Рассчитать вероятности"):
        try:
            team_A_id = int(team_A_id)
            team_B_id = int(team_B_id)
        except ValueError:
            st.error("Пожалуйста, введите числовые значения для ID команд.")
            return

        result = predict_match_outcome(df_game, team_A_id, team_B_id, recent_games)
        
        # Преобразуем столбец с датой, чтобы оставить только дату (если это необходимо)
        team_A_recent = result.get("recent_games_info").get("team_A").copy()
        if "date" in team_A_recent.columns:
            team_A_recent["date"] = pd.to_datetime(team_A_recent["date"], errors='coerce').dt.date
        team_A_recent = team_A_recent.rename(columns=mapping)

        # Аналогично для команды B
        team_B_recent = result.get("recent_games_info").get("team_B").copy()
        if "date" in team_B_recent.columns:
            team_B_recent["date"] = pd.to_datetime(team_B_recent["date"], errors='coerce').dt.date
        team_B_recent = team_B_recent.rename(columns=mapping)

        # Если недостаточно данных для расчета
        if result.get("team_A_win_prob") is None:
            st.error(result.get("message"))
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Команда A ({team_A_name})")
                st.write(f"Всего игр: {result.get('team_A_games_count')}")
                st.write(f"Рейтинг: {result.get('team_A_rating')}")
                st.write(f"Вероятность победы: {result.get('team_A_win_prob')}")
                st.write("Вероятность победы за всю историю игр:", result.get("overall_probs").get("team_A"))
                st.write("Вероятность победы за последние игры:", result.get("recent_probs").get("team_A"))
                st.write("Последние игры:")
                st.dataframe(team_A_recent.reset_index(drop=True))
                
            with col2:
                st.subheader(f"Команда B ({team_B_name})")
                st.write(f"Всего игр: {result.get('team_B_games_count')}")
                st.write(f"Рейтинг: {result.get('team_B_rating')}")
                st.write(f"Вероятность победы: {result.get('team_B_win_prob')}")
                st.write("Вероятность победы за всю историю игр:", result.get("overall_probs").get("team_B"))
                st.write("Вероятность победы за последние игры:", result.get("recent_probs").get("team_B"))
                st.write("Последние игры:")
                st.dataframe(team_B_recent.reset_index(drop=True))
