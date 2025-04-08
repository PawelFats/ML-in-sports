import pandas as pd

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
    
    # Вывод количества игр для отладки
    print(f"A games: {games_count_A}")
    print(f"B games: {games_count_B}")
    
    return {
        "team_A_win_prob": round(norm_A, 2),
        "team_B_win_prob": round(norm_B, 2),
        "overall_probs": {"team_A": p_A_overall, "team_B": p_B_overall},
        "recent_probs": {"team_A": p_A_recent, "team_B": p_B_recent},
        "recent_games_info": {
            "team_A": recent_info_A,
            "team_B": recent_info_B
        }
    }

