DATA_PATH = "data/"
HISTORY_FILE = DATA_PATH + "history.csv"
CACHE_TTL = 3600  # сколько секунд держать данные в памяти
METRICS = ['goals', 'assists', 'throws_by', 'shot_on_target', 'blocked_throws', 'p_m']
P_METRICS = ['p_goals', 'p_assists', 'p_throws_by', 'p_shot_on_target', 'p_blocked_throws', 'p_p_m']
COEFFICIENTS = {"defender": 2/3, "forward": 1/6}
