import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import json
import os

METRICS = ['goals', 'assists', 'assists_2', 'throws_by', 'shot_on_target', 'blocked_throws', 'p_m']
P_METRICS = [f'p_{m}' for m in METRICS]

def build_division_weights_ui(df_history: pd.DataFrame, defaults: dict[int, float] | None = None) -> dict[int, float]:
    """UI блок с весами дивизионов (можно скрыть/раскрыть)."""
    unique_divisions = sorted(pd.Series(df_history.get('division', pd.Series(dtype=float))).dropna().unique())
    division_weights: dict[int, float] = {}
    with st.expander("Веса дивизионов", expanded=False):
        for div in unique_divisions:
            try:
                div_int = int(div)
            except Exception:
                continue
            division_weights[div_int] = st.slider(
                f"Дивизион {div_int}", 0.0, 3.0, float((defaults or {}).get(div_int, 1.0)), step=0.1, key=f"div_w_{div_int}"
            )
    return division_weights

def build_amplua_weights_ui(defaults: dict[str, float] | None = None) -> tuple[float, float, float]:
    """UI блок с весами амплуа (можно скрыть/раскрыть)."""
    with st.expander("Веса амплуа", expanded=False):
        coef_def = st.slider("Вес защитников", 0.0, 3.0, float((defaults or {}).get('def', 1.0)), step=0.1)
        coef_att = st.slider("Вес нападающих", 0.0, 3.0, float((defaults or {}).get('att', 1.0)), step=0.1)
        coef_gk  = st.slider("Вес вратаря",    0.0, 3.0, float((defaults or {}).get('gk', 1.0)),  step=0.1)
    return coef_def, coef_att, coef_gk

def build_metric_weights_ui(defaults_from_file: dict[str, float] | None = None) -> dict[str, float]:
    """UI блок с весами показателей (можно скрыть/раскрыть)."""
    defaults = {
        'goals': 1.0,
        'assists': 0.8,
        'assists_2': 0.6,
        'throws_by': 0.1,
        'shot_on_target': 0.3,
        'blocked_throws': 0.5,
        'p_m': 0.5,
    }
    if defaults_from_file:
        defaults.update({k: float(v) for k, v in defaults_from_file.items() if k in defaults})
    weights: dict[str, float] = {}
    with st.expander("Веса показателей", expanded=False):
        for metric in METRICS:
            label_ru = {
                'goals': 'голы',
                'assists': 'ассисты',
                'assists_2': 'ассисты_2',
                'throws_by': 'броски мимо',
                'shot_on_target': 'броски в створ',
                'blocked_throws': 'блокированные броски',
                'p_m': 'п/м',
            }.get(metric, metric)
            weights[metric] = st.slider(
                f"Вес {label_ru}", 0.0, 3.0, float(defaults.get(metric, 1.0)), step=0.1, key=f"m_w_{metric}"
            )
    return weights

WEIGHTS_FILE = r"data/processed/red_method/weights_red.json"

def load_saved_weights() -> dict:
    try:
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Приводим ключи дивизионов к int
                divs = data.get('divisions', {})
                data['divisions'] = {int(k): float(v) for k, v in divs.items()}
                return data
    except Exception:
        pass
    return {
        'amplua': {'def': 1.0, 'att': 1.0, 'gk': 1.0},
        'metrics': {},
        'divisions': {},
        'goalie_metrics': {'%': 1.0, 'КН': 1.0, 'СИ': 1.0, 'П': 1.0}
    }

def save_weights(amplua: tuple[float, float], metrics: dict[str, float], divisions: dict[int, float], goalie_metrics: dict[str, float] | None = None) -> None:
    payload = {
        'amplua': {'def': float(amplua[0]), 'att': float(amplua[1]), 'gk': float(amplua[2])},
        'metrics': {k: float(v) for k, v in metrics.items()},
        'divisions': {str(int(k)): float(v) for k, v in divisions.items()},
        'goalie_metrics': {k: float(v) for k, v in (goalie_metrics or {}).items()},
    }
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def compute_latest_player_division(df_compile: pd.DataFrame, df_history: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками ['ID player', 'division'] — последний известный дивизион игрока по дате игры.
    """
    if 'ID game' not in df_compile.columns:
        return pd.DataFrame(columns=['ID player', 'division'])
    hist_cols = [c for c in ['ID', 'date', 'division'] if c in df_history.columns]
    merged = pd.merge(
        df_compile[['ID player', 'ID game']].drop_duplicates(),
        df_history[hist_cols],
        left_on='ID game', right_on='ID', how='left'
    )
    if 'date' in merged.columns:
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        merged = merged.sort_values('date')
    merged = merged.dropna(subset=['division']) if 'division' in merged.columns else merged
    latest = merged.drop_duplicates(subset=['ID player'], keep='last')
    return latest[['ID player', 'division']] if {'ID player', 'division'}.issubset(latest.columns) else pd.DataFrame(columns=['ID player', 'division'])

def compute_goalkeepers_ratings(
    df_goalkeepers: pd.DataFrame,
    goals_and_passes_path: str = r"data/raw/goals_and_passes.csv",
    goalie_metric_weights: dict[str, float] | None = None,
    allowed_game_ids: set[int] | list[int] | None = None,
    allowed_team_ids: set[int] | list[int] | None = None,
    include_team_col: bool = False,
    amplua_weight_gk: float = 1.0,
) -> pd.DataFrame:
    """
    Формирует агрегированную таблицу рейтингов вратарей для красного метода.
    Выходные колонки: ['ID player', '%', 'КН', 'СИ', 'П']
    Где:
      - %: общий процент отраженных бросков = (1 - sum(missed pucks)/sum(total throws)) * 100
      - КН: коэффициент надежности = (60 * sum(missed pucks)) / (60 * games) = sum(missed pucks) / games
      - СИ: число сухих игр = count(missed pucks == 0)
      - П: победы = число игр, где команда вратаря забила больше соперника
    """
    if df_goalkeepers is None or df_goalkeepers.empty:
        return pd.DataFrame(columns=['ID player', '%', 'КН', 'СИ', 'П'])

    # Приводим названия колонок к ожидаемым
    cols_map = {
        'ID game': 'ID game',
        'ID team': 'ID team',
        'ID player': 'ID player',
        'missed pucks': 'missed pucks',
        'total throws': 'total throws',
        '% of reflected shots': '% of reflected shots',
    }
    for c in cols_map:
        if c not in df_goalkeepers.columns:
            # Если что-то критичное отсутствует
            return pd.DataFrame(columns=['ID player', '%', 'КН', 'СИ', 'П'])

    df_gk = df_goalkeepers.copy()
    # Фильтры по играм/командам
    if allowed_game_ids is not None:
        allowed_game_ids = set(int(x) for x in allowed_game_ids)
        df_gk = df_gk[df_gk['ID game'].astype(int).isin(allowed_game_ids)]
    if allowed_team_ids is not None:
        allowed_team_ids = set(int(x) for x in allowed_team_ids)
        df_gk = df_gk[df_gk['ID team'].astype(int).isin(allowed_team_ids)]
    # Убедимся, что числовые поля приведены
    df_gk['missed pucks'] = pd.to_numeric(df_gk['missed pucks'], errors='coerce').fillna(0)
    df_gk['total throws'] = pd.to_numeric(df_gk['total throws'], errors='coerce').fillna(0)

    # Победы: считаем голы по данным goals_and_passes
    try:
        df_gap = pd.read_csv(goals_and_passes_path)
        # Проверим наличие необходимых колонок
        if not {'ID game', 'ID team'}.issubset(df_gap.columns):
            raise KeyError('goals_and_passes missing required columns')
        if allowed_game_ids is not None and len(allowed_game_ids) > 0:
            df_gap = df_gap[df_gap['ID game'].astype(int).isin(allowed_game_ids)]
        # Подсчет голов по игре и команде
        goals_by_team = df_gap.groupby(['ID game', 'ID team']).size().reset_index(name='team_goals')
        goals_total = df_gap.groupby(['ID game']).size().reset_index(name='total_goals')
        # Объединим с уникальными строками вратарей по игре и команде
        gk_games = df_gk[['ID game', 'ID team', 'ID player']].drop_duplicates()
        gk_games = gk_games.merge(goals_by_team, on=['ID game', 'ID team'], how='left')
        gk_games = gk_games.merge(goals_total, on='ID game', how='left')
        gk_games['team_goals'] = gk_games['team_goals'].fillna(0)
        gk_games['total_goals'] = gk_games['total_goals'].fillna(0)
        gk_games['opponent_goals'] = (gk_games['total_goals'] - gk_games['team_goals']).clip(lower=0)
        gk_games['win'] = (gk_games['team_goals'] > gk_games['opponent_goals']).astype(int)
        wins_by_player = gk_games.groupby('ID player')['win'].sum().reset_index(name='П')
    except Exception:
        # Если файл не загрузился, проставим нули
        wins_by_player = df_gk.groupby('ID player').size().reset_index(name='П')
        wins_by_player['П'] = 0

    # Агрегации по игроку: суммарные броски, пропущенные, игры, сухие матчи
    group_keys = ['ID player', 'ID team'] if include_team_col else ['ID player']
    agg = df_gk.groupby(group_keys).agg(
        sum_missed=('missed pucks', 'sum'),
        sum_throws=('total throws', 'sum'),
        games=('ID game', 'nunique'),
        clean_sheets=('missed pucks', lambda s: int((s == 0).sum())),
    ).reset_index()

    # % отраженных бросков
    agg['%'] = np.where(agg['sum_throws'] > 0, (1 - agg['sum_missed'] / agg['sum_throws']) * 100.0, np.nan)
    # КН по формуле (60*пропущенные)/(60*игры) = пропущенные/игры
    agg['КН'] = np.where(agg['games'] > 0, agg['sum_missed'] / agg['games'], np.nan)
    agg['СИ'] = agg['clean_sheets']

    # Победы
    # Победы: считаем по тем же ключам группировки
    if include_team_col and 'ID team' in gk_games.columns:
        wins_by = gk_games.groupby(['ID player', 'ID team'])['win'].sum().reset_index(name='П')
        agg = agg.merge(wins_by, on=['ID player', 'ID team'], how='left')
    else:
        wins_by = gk_games.groupby(['ID player'])['win'].sum().reset_index(name='П')
        agg = agg.merge(wins_by, on=['ID player'], how='left')
    agg['П'] = agg['П'].fillna(0).astype(int)

    # Финальные колонки (база)
    base_cols = ['ID player', 'ID team'] if include_team_col else ['ID player']
    result = agg[base_cols + ['%', 'КН', 'СИ', 'П', 'games']].copy()
    result['%'] = result['%'].round(2)
    result['КН'] = result['КН'].round(2)

    # Весовые очки для вратарских метрик
    weights = {
        '%': 1.0,
        'КН': 1.0,
        'СИ': 1.0,
        'П': 1.0,
    }
    if goalie_metric_weights:
        # ожидаем ключи: '%', 'КН', 'СИ', 'П'
        for k in list(weights.keys()):
            if k in goalie_metric_weights:
                try:
                    weights[k] = float(goalie_metric_weights[k])
                except Exception:
                    pass

    # Преобразование КН (чем меньше, тем лучше) → "чем больше, тем лучше"
    max_kn = result['КН'].max(skipna=True)
    if pd.isna(max_kn):
        max_kn = 0.0
    kn_eff = (max_kn - result['КН']).clip(lower=0)

    # Очки по метрикам: по аналогии с полевыми ((stat + coef_gk * games)^2) / games с учетом веса
    g = result['games'].replace(0, np.nan)
    result['%_о'] = (weights['%'] * ((result['%'] + amplua_weight_gk * g) ** 2) / g).fillna(0.0)
    result['КН_о'] = (weights['КН'] * ((kn_eff + amplua_weight_gk * g) ** 2) / g).fillna(0.0)
    result['СИ_о'] = (weights['СИ'] * ((result['СИ'] + amplua_weight_gk * g) ** 2) / g).fillna(0.0)
    result['П_о'] = (weights['П'] * ((result['П'] + amplua_weight_gk * g) ** 2) / g).fillna(0.0)

    # Итоговый рейтинг
    result['рейтинг'] = (result['%_о'] + result['КН_о'] + result['СИ_о'] + result['П_о']).round(2)

    # Возвращаем без служебных колонок
    # Переименуем games в 'И' для вывода
    result.rename(columns={'games': 'И'}, inplace=True)
    ordered = base_cols + ['И', '%', 'КН', 'СИ', 'П', '%_о', 'КН_о', 'СИ_о', 'П_о', 'рейтинг']
    return result[ordered]

@st.cache_data
def load_data():
    df_history = pd.read_csv(r"data/raw/game_history.csv")
    df_compile_stats = pd.read_csv(r'data/targeted/compile_stats.csv')
    df_goalk_stats = pd.read_csv(r'data/targeted/goalkeepers_data.csv')
    return df_history, df_compile_stats, df_goalk_stats

def rename_columns(df):
    """
    Переименовывает столбцы датафрейма в соответствии с заданной схемой.
    
    Если в датафрейме присутствует один из ключевых столбцов, он будет заменён на
    соответствующее название на русском языке.
    
    Аргументы:
        df (pandas.DataFrame): Исходный датафрейм с английскими названиями столбцов.
        
    Возвращает:
        pandas.DataFrame: Датафрейм с переименованными столбцами.
    """
    mapping = {
        'ID player': 'ID игрока',
        'amplua': 'амплуа',
        'games': 'игры',
        'goals': 'шайбы',
        'p_goals': 'о_ш',
        'assists': 'ассисты',
        'p_assists': 'о_а',
        'assists_2': 'ассисты_2',
        'p_assists_2': 'о_а_2',
        'throws_by': 'броски_мимо',
        'p_throws_by': 'о_м',
        'shot_on_target': 'броски_в_створ',
        'p_shot_on_target': 'о_с',
        'blocked_throws': 'блок_броски',
        'p_blocked_throws': 'о_б',
        'p_m': 'п/м',
        'p_p_m': 'о_п/м',
        'player_rating': 'общий_рейтинг'
    }
    # Переименовываем столбцы, если они присутствуют в датафрейме
    df = df.rename(columns=mapping)
    return df


def calculate_player_stats(df, output_file=r"data/processed/red_method/player_stats.csv"):
    """
    Считает суммарные достижения для каждого игрока за всё время.
    """
    df_filtered = df[df['amplua'].isin([9, 10])]
    
    player_stats = df_filtered.groupby(['ID player', 'amplua']).agg(
        games=('ID game', 'nunique'),
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        assists_2=('assists_2', 'sum'),
        throws_by=('throws by', 'sum'),
        shot_on_target=('a shot on target', 'sum'),
        blocked_throws=('blocked throws', 'sum'),
        p_m=('p/m', 'sum'),
        #time=('total time on ice', 'sum')
    ).reset_index()

    player_stats.to_csv(output_file, index=False)
    
    return player_stats

# Обновлённая сигнатура calculate_points
def calculate_points(df: pd.DataFrame,
                          coefficient: float,
                          amplua: int,
                          metric_weights: dict[str, float]) -> pd.DataFrame:
    """
    Вычисляет взвешенные p_метрики и итоговый рейтинг для заданного амплуа.
    :param df: входные статистики игроков
    :param coefficient: коэффициент амплуа (защитники/нападающие)
    :param amplua: код амплуа (9 — защитники, 10 — нападающие)
    :param metric_weights: словарь весов для каждой метрики
    """
    df_filtered = df[df['amplua'] == amplua].copy()


    # Расчёт взвешенных p_метрик
    for col in METRICS:
        raw = ((df_filtered[col] + coefficient * df_filtered['games']) ** 2) / df_filtered['games']
        weight = metric_weights.get(col, 1.0)
        df_filtered[f'p_{col}'] = weight * raw

    # Итоговый рейтинг — сумма всех взвешенных p_метрик
    p_cols = [f'p_{c}' for c in METRICS]
    df_filtered['player_rating'] = df_filtered[p_cols].sum(axis=1)

    return df_filtered

# Обновлённая функция process_and_save
def process_and_save(
        df: pd.DataFrame,
        coef_def: float,
        coef_att: float,
        metric_weights: dict[str, float],
        division_weights: dict[int, float] | None = None,
        output_file: str = r"data/processed/red_method/player_stats_with_points.csv"
 ) -> pd.DataFrame:
    """
    Рассчитывает и сохраняет актуальные рейтинги игроков с учётом
    пользовательских весов амплуа и метрик.
    :param df: исходные сырые данные compile_stats
    :param coef_def: коэффициент для защитников
    :param coef_att: коэффициент для нападающих
    :param metric_weights: словарь весов по каждому показателю
    :param output_file: путь для сохранения CSV
    """
    # Собираем базовые агрегированные статистики
    df_stats = calculate_player_stats(df)

    # Применяем расчёт очков с пользовательскими весами
    df_defenders = calculate_points(df_stats, coef_def, 9, metric_weights)
    df_forwards  = calculate_points(df_stats, coef_att,   10, metric_weights)

    # Объединяем результаты
    df_final = pd.concat([df_defenders, df_forwards], ignore_index=True)

    # Применяем вес дивизиона ПОСЛЕ расчёта рейтинга игрока
    if division_weights:
        # Вычисляем последний известный дивизион игрока
        latest_player_div = compute_latest_player_division(df, pd.read_csv(r"data/raw/game_history.csv"))
        df_final = pd.merge(df_final, latest_player_div, on='ID player', how='left')
        df_final['division'] = df_final['division'].astype('Int64')
        df_final['player_rating'] = df_final.apply(
            lambda r: r['player_rating'] * float(division_weights.get(int(r['division']) if pd.notna(r['division']) else 0, 1.0)),
            axis=1
        )

    # Упорядочиваем колонки по образцу
    expected_cols = [
        'ID player', 'amplua', 'games',
        'goals', 'p_goals',
        'assists', 'p_assists',
        'assists_2', 'p_assists_2',
        'throws_by', 'p_throws_by',
        'shot_on_target', 'p_shot_on_target',
        'blocked_throws', 'p_blocked_throws',
        'p/m', 'p_p_m',
        'player_rating'
    ]
    cols = [c for c in expected_cols if c in df_final.columns]
    df_final = df_final[cols]

    # Сохраняем результат
    df_final.to_csv(output_file, index=False)

    return df_final

# Обновлённая функция process_seasons
def process_seasons(
         df_compile: pd.DataFrame,
         df_history: pd.DataFrame,
         season_ids: list[int] | None,
         player_ids: list[int] | None,
         coef_def: float,
         coef_att: float,
         metric_weights: dict[str, float],
          division_weights: dict[int, float] | None = None,
         output_file: str = r"data/processed/red_method/season_player_stats_with_points.csv"
 ) -> pd.DataFrame:
    """
    1. Фильтрует игры по списку сезонов и, при необходимости, по выбранным игрокам.
    2. Выводит статистику игр, команд и игроков.
    3. Суммирует несколько сезонов и вызывает process_and_save.
    """
    if season_ids:
        df_hist = df_history[df_history["ID season"].isin([int(s) for s in season_ids])]
    else:
        df_hist = df_history.copy()

    df_season = pd.merge(
        df_compile,
        df_hist[["ID"]],
        left_on="ID game",
        right_on="ID",
        how="inner"
    )
    if player_ids:
        df_season = df_season[df_season["ID player"].isin([int(pid) for pid in player_ids])]

    unique_games = df_season['ID game'].nunique()
    unique_teams = df_season['ID team'].nunique()
    unique_players = df_season.groupby('amplua')['ID player'].nunique()

    st.write(f"Уникальных игр в выбранных сезонах: {unique_games}")
    st.write(f"Уникальных команд в выбранных сезонах: {unique_teams}")
    st.write("Уникальных игроков по амплуа:")
    for amp, cnt in unique_players.items():
        label = {8:'Вратари',9:'Защитники',10:'Атакующие'}.get(amp, amp)
        st.write(f"  {label}: {cnt}")

    # Запускаем перерасчёт с пользовательскими весами
    df_final = process_and_save(
        df_season,
        coef_def,
        coef_att,
        metric_weights,
        division_weights,
        output_file
    )
    return df_final

def plot_player_ratings(result_df, season_id):
    """
    Отрисовывает два графика:
      - Составной столбчатый график рейтингов по показателям.
      - График количества игр.
    """
    metric_labels = {
        'p_goals': 'голы',
        'p_assists': 'ассисты',
        'p_assists_2': 'ассисты_2',
        'p_throws_by': 'мимо',
        'p_shot_on_target': 'в створ',
        'p_blocked_throws': 'блокированные',
        'p_p_m': 'п/м'
    }
    
    players = result_df['ID player'].astype(str)
    metrics = list(metric_labels.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # 1. График: Составной столбчатый график (stacked bar chart) рейтингов в процентах
    # Рассчитываем процентное соотношение каждого показателя
    result_df_percent = result_df.copy()

    for metric in metrics:
        result_df_percent[metric] = (result_df_percent[metric] / result_df_percent['player_rating']) * 100

    fig1, ax1 = plt.subplots(figsize=(14,8))
    bottom = np.zeros(len(result_df_percent))
    for i, metric in enumerate(metrics):
        ax1.bar(players, result_df_percent[metric], bottom=bottom, color=colors[i], label=metric_labels[metric])
        bottom += result_df_percent[metric].values
    ax1.set_xlabel('ID игрока')
    ax1.set_ylabel('Рейтинговые очки')
    ax1.set_title(f'Структура рейтингов по показателям за сезон {season_id}')
    ax1.legend(title='Показатели', bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # График 2: количество игр
    fig2, ax2 = plt.subplots(figsize=(14,6))
    ax2.bar(players, result_df['games'], color='skyblue')
    ax2.set_xlabel('ID игрока')
    ax2.set_ylabel('Количество игр')
    ax2.set_title(f'Количество игр за сезон {season_id}')
    plt.xticks(rotation=45)

    # 3. Тепловая карта показателей
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    heatmap_data = result_df.set_index('ID player')[metrics].rename(columns=metric_labels)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={'label': 'Рейтинговые очки'},
        ax=ax3
    )
    ax3.set_title(f'Тепловая карта показателей за сезон {season_id}\n', fontsize=16, pad=20)
    ax3.set_xlabel('Показатели', fontsize=12)
    ax3.set_ylabel('ID игрока', fontsize=12)
    
    # 4. Группированный бар-чарт по метрикам
    fig4, ax4 = plt.subplots(figsize=(16, 8))
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
        edgecolor='w',
        ax=ax4
    )
    ax4.set_title(f'Распределение показателей по игрокам (сезон {season_id})', fontsize=16)
    ax4.set_xlabel('ID игрока', fontsize=12)
    ax4.set_ylabel('Рейтинговые очки', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Показатели', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Радарная диаграмма для топ-5 игроков
    #result_df['total_rating'] = result_df[metrics].sum(axis=1)
    top_players = result_df.nlargest(5, 'player_rating') if not result_df.empty else result_df
    labels = list(metric_labels.values())
    num_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    radar_colors = sns.color_palette("husl", 5)
    fig5, ax5 = plt.subplots(subplot_kw={'polar': True}, figsize=(10, 10))
    
    top_players = top_players.reset_index(drop=True)
    for i, (_, row) in enumerate(top_players.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]
        ax5.plot(angles, values, color=radar_colors[i], linewidth=2, 
                 label=f"ID {row['ID player']} (Σ={row['player_rating']:.1f})")
        ax5.fill(angles, values, color=radar_colors[i], alpha=0.1)
    
    ax5.set_theta_offset(np.pi / 2)
    ax5.set_theta_direction(-1)
    ax5.set_rlabel_position(30)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(labels, fontsize=10)
    ax5.tick_params(axis='y', labelsize=8)
    ax5.set_title('Топ-5 игроков по суммарному рейтингу\n', fontsize=16, pad=40)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10, frameon=True, shadow=True)
    ax5.spines['polar'].set_visible(False)
    ax5.grid(alpha=0.5, linestyle='--')
    
    # Возвращаем все созданные фигуры
    return fig1, fig2, fig3, fig4, fig5

def plot_team_ratings(
         df_compile: pd.DataFrame,
         df_history: pd.DataFrame,
         season_id: int|list[int]|None,
         team_ids: list[int]|None,
         coef_def: float,
         coef_att: float,
         metric_weights: dict[str, float],
         division_weights: dict[int, float] | None = None
 ):
    """
    Функция строит рейтинги команд с учётом пользовательских весов.
    """
    # 1. Фильтрация по сезонам
    if season_id is not None:
        if isinstance(season_id, (list, tuple)):
            seasons = [int(s) for s in season_id]
            df_hist = df_history[df_history["ID season"].isin(seasons)]
        else:
            df_hist = df_history[df_history["ID season"] == int(season_id)]
        df = pd.merge(df_compile, df_hist[["ID","division"]],
                      left_on="ID game", right_on="ID", how="inner")
    else:
        df = pd.merge(df_compile, df_history[["ID","division"]],
                      left_on="ID game", right_on="ID", how="left")

    # 2. Фильтрация по выбранным командам
    if team_ids:
        df = df[df["ID team"].isin(team_ids)]
        
    # 3. Агрегация по игрокам
    df_grouped = df.groupby(['ID player','amplua','ID team']).agg(
        games=('ID game','nunique'),
        goals=('goals','sum'),
        assists=('assists','sum'),
        assists_2=('assists_2','sum'),
        throws_by=('throws by','sum'),
        shot_on_target=('a shot on target','sum'),
        blocked_throws=('blocked throws','sum'),
        p_m=('p/m','sum')
    ).reset_index()

    # 4. Расчёт очков с учётом весов
    df_def = calculate_points(df_grouped, coef_def, 9, metric_weights)
    df_att = calculate_points(df_grouped, coef_att,10, metric_weights)
    df_players = pd.concat([df_def, df_att], ignore_index=True)

    # 5. Суммарный рейтинг команд
    df_team = df_players.groupby('ID team',as_index=False).agg(
        team_rating=('player_rating','sum')
    )
    df_team['team_rating'] = df_team['team_rating'].round(2)
    divs = df.drop_duplicates(subset='ID team')[['ID team','division']]
    df_team = df_team.merge(divs,on='ID team',how='left')

    # Применяем вес дивизиона к ИТОГОВОМУ рейтингу команды
    if division_weights:
        df_team['division'] = df_team['division'].astype('Int64')
        df_team['team_rating'] = df_team.apply(
            lambda r: r['team_rating'] * float(division_weights.get(int(r['division']) if pd.notna(r['division']) else 0, 1.0)),
            axis=1
        )
        df_team['team_rating'] = df_team['team_rating'].round(2)
    
    # 1. Столбчатая диаграмма суммарного рейтинга для команд
    fig_total, ax_total = plt.subplots(figsize=(12, 7))
    team_labels = df_team['ID team'].astype(str) + ' (Div ' + df_team['division'].astype(str) + ')'
    bars = ax_total.bar(team_labels, df_team['team_rating'], color='teal')
    if season_id is not None:
        ax_total.set_title(f'Суммарный рейтинг команд в сезоне {season_id}', fontsize=14)
    else:
        ax_total.set_title('Суммарный рейтинг команд за всё время', fontsize=14)
    ax_total.set_xlabel('ID команды', fontsize=12)
    ax_total.set_ylabel('Суммарный рейтинг', fontsize=12)
    plt.xticks(rotation=45)
    for bar, team_rating in zip(bars, df_team['team_rating']):
        height = bar.get_height()
        ax_total.text(bar.get_x() + bar.get_width() / 2, height, team_rating,
                      ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    
    # 2. Составной (stacked) бар-чарт с разбивкой по амплуа
    # Группируем по командам и амплуа, суммируя рейтинговые очки
    df_team_amplua = df_players.groupby(['ID team', 'amplua'])['player_rating'].sum().unstack(fill_value=0).reset_index()
    # Добавляем информацию о дивизионе
    df_team_amplua = pd.merge(df_team_amplua, divs, on='ID team', how='left')
    team_labels_stacked = df_team_amplua['ID team'].astype(str) + ' (Div ' + df_team_amplua['division'].astype(str) + ')'
    
    fig_stacked, ax_stacked = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(df_team_amplua))
    # Определяем цвета и подписи для амплуа
    amplua_colors = {9: 'skyblue', 10: 'salmon'}
    amplua_labels = {9: 'Защитники', 10: 'Атакующие'}
    for amplua in [9, 10]:
        if amplua in df_team_amplua.columns:
            values = df_team_amplua[amplua]
            ax_stacked.bar(team_labels_stacked, values, bottom=bottom, 
                           color=amplua_colors[amplua], label=amplua_labels[amplua])
            bottom += values.values
    ax_stacked.set_title('Структура рейтинга по амплуа', fontsize=14)
    ax_stacked.set_xlabel('Команда', fontsize=12)
    ax_stacked.set_ylabel('Суммарный рейтинг', fontsize=12)
    ax_stacked.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 3. Диаграмма рассеяния: зависимость количества уникальных игроков от суммарного рейтинга команды
    # Считаем число уникальных игроков для каждой команды
    df_team_players = df_players.groupby('ID team')['ID player'].nunique().reset_index(name='num_players')
    df_scatter = pd.merge(df_team, df_team_players, on='ID team', how='left')
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 7))
    ax_scatter.scatter(df_scatter['team_rating'], df_scatter['num_players'], color='purple', s=100, alpha=0.7)
    for idx, row in df_scatter.iterrows():
        ax_scatter.text(row['team_rating'], row['num_players'], str(row['ID team']),
                        fontsize=9, ha='center', va='bottom')
    ax_scatter.set_xlabel("Суммарный рейтинг команды", fontsize=12)
    ax_scatter.set_ylabel("Количество уникальных игроков", fontsize=12)
    ax_scatter.set_title("Зависимость числа игроков от рейтинга команды", fontsize=14)
    
    plt.tight_layout()
    
    # 4. Составной бар-чарт с процентным соотношением вклада показателей в общий рейтинг команды
    # Определяем метрики и их подписи
    metric_labels = {
        'p_goals': 'голы',
        'p_assists': 'ассисты',
        'p_assists_2': 'ассисты_2',
        'p_throws_by': 'мимо',
        'p_shot_on_target': 'в створ',
        'p_blocked_throws': 'блокированные',
        'p_p_m': 'п/м'
    }
    metrics = list(metric_labels.keys())
    
    # Группируем данные по командам для суммирования метрик
    df_team_metrics = df_players.groupby('ID team')[metrics].sum().reset_index()
    # Рассчитываем общую сумму метрик для каждой команды
    df_team_metrics['total'] = df_team_metrics[metrics].sum(axis=1)
    # Переводим каждую метрику в проценты от общего рейтинга
    for metric in metrics:
        df_team_metrics[metric] = df_team_metrics[metric] / df_team_metrics['total'] * 100
    
    # Добавляем информацию о дивизионе
    df_team_metrics = pd.merge(df_team_metrics, divs, on='ID team', how='left')
    team_labels_metric = df_team_metrics['ID team'].astype(str) + ' (Div ' + df_team_metrics['division'].astype(str) + ')'
    
    fig_metric, ax_metric = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(df_team_metrics))
    colors_metric = {
    'p_goals': '#1f77b4',
    'p_assists': '#ff7f0e',
    'p_assists_2': '#2ca02c',
    'p_throws_by': '#d62728',
    'p_shot_on_target': '#9467bd',
    'p_blocked_throws': '#8c564b',
    'p_p_m': '#e377c2'
    }

    
    for metric in metrics:
        values = df_team_metrics[metric]
        ax_metric.bar(team_labels_metric, values, bottom=bottom, 
                      color=colors_metric[metric], label=metric_labels[metric])
        bottom += values.values
    ax_metric.set_title("Процентное соотношение показателей в общем рейтинге команды", fontsize=14)
    ax_metric.set_xlabel("ID команды", fontsize=12)
    ax_metric.set_ylabel("Процентный вклад", fontsize=12)
    ax_metric.legend(title='Показатели', bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 7. Радарная диаграмма для топ-5 команд по суммарному рейтингу
    df_radar = df_players.groupby('ID team')[['p_goals','p_assists', 'p_assists_2', 'p_throws_by','p_shot_on_target','p_blocked_throws','p_p_m','player_rating']].sum().reset_index()
    top_teams = df_radar.nlargest(5, 'player_rating')  # здесь можно выбрать любую метрику или суммарный рейтинг
    labels = ['голы', 'ассисты', 'ассисты_2', 'мимо', 'в створ', 'блокированные', 'п/м']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # замыкаем круг

    fig_radar, ax_radar = plt.subplots(subplot_kw={'polar': True}, figsize=(10, 10))
    radar_colors = sns.color_palette("Set2", n_colors=top_teams.shape[0])
    top_teams = top_teams.reset_index(drop=True)

    for i, row in top_teams.iterrows():
        values = row[P_METRICS].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, color=radar_colors[i], linewidth=2, label=f"Команда ID {row['ID team']} ({round(row['player_rating'], 2)})")
        ax_radar.fill(angles, values, color=radar_colors[i], alpha=0.25)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax_radar.set_title("Радарная диаграмма для топ-5 команд", fontsize=14)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Возвращаем таблицу и все созданные графики
    return df_players, df_team, fig_total, fig_stacked, fig_scatter, fig_metric, fig_radar

def player_rt_red():
    # Заголовок
    st.title("Рейтинг игроков (советский метод)")

    # Загружаем сохранённые веса
    saved = load_saved_weights()

    # Постоянные UI-блоки под заголовком
    coef_def, coef_att, coef_gk = build_amplua_weights_ui(saved.get('amplua'))
    metric_weights = build_metric_weights_ui(saved.get('metrics'))
    
    with st.spinner("Загрузка данных..."):
        df_history, df_compile_stats, df_goalk_stats = load_data()
    
    # Объединяем данные, чтобы брать только те сезоны, в которых есть игры из compile_stats
    df_merged = pd.merge(df_compile_stats, df_history, left_on="ID game", right_on="ID", how="inner")
    available_seasons = sorted(df_merged["ID season"].unique())

    # Блок весов дивизионов + блок весов вратарей (после загрузки данных)
    division_weights = build_division_weights_ui(df_history, saved.get('divisions'))
    with st.expander("Веса показателей вратарей", expanded=False):
        goalie_metric_weights = {}
        for k, label in [('%', '% отраженных'), ('КН', 'КН (меньше — лучше)'), ('СИ', 'Сухие игры'), ('П', 'Победы')]:
            default_val = float(saved.get('goalie_metrics', {}).get(k, 1.0))
            goalie_metric_weights[k] = st.slider(f"Вес {label}", 0.0, 5.0, default_val, step=0.01, key=f"gk_w_{k}")

    # Кнопка сохранения весов
    if st.button("Сохранить веса"):
        # goalie_metric_weights определен выше при построении блока весов вратарей
        save_weights((coef_def, coef_att, coef_gk), metric_weights, division_weights, goalie_metric_weights)
        st.success("Веса сохранены и будут подставляться по умолчанию")
    
    action = st.selectbox(
        "Выберите действие",
        ["Актуальный рейтинг игроков", "Сезонная статистика игроков", "Сезонная статистика команд", "Рейтинг вратарей"]
    )

    
    # 1. Общая статистика игроков
    if action == "Актуальный рейтинг игроков":
        stats = process_and_save(
            df_compile_stats,
            coef_def,
            coef_att,
            metric_weights,
            division_weights,
        )
        stats = rename_columns(stats)
        st.dataframe(stats)

    
    # 2. Статистика за сезоны
    elif action == "Сезонная статистика игроков":
        # Мультивыбор сезонов
        season_ids = st.multiselect("Выберите сезоны", options=available_seasons, default=available_seasons[:2])
        
        # Фильтруем DataFrame по выбранным сезонам и исключаем amplua == 8
        df_filtered = df_merged[df_merged["amplua"] != 8]
        if season_ids:
            df_filtered = df_filtered[df_filtered["ID season"].isin(season_ids)]

        # Уникальные игроки по амплуа
        players_9 = df_filtered[df_filtered["amplua"] == 9]["ID player"].unique()
        players_10 = df_filtered[df_filtered["amplua"] == 10]["ID player"].unique()

        st.write("Выберите игроков или оставьте поля пустыми:")
        players_input_9 = st.multiselect("Защитники", options=players_9, default=players_9[:2])
        players_input_10 = st.multiselect("Атакующие", options=players_10, default=players_10[:2])
        players_input = list(players_input_9) + list(players_input_10)

        # Проверяем, есть ли выбранные сезоны или игроки
        has_seasons = len(season_ids) > 0
        has_players = len(players_input) > 0
        
        if not has_seasons and not has_players:
            st.info("⚠️ Выберите хотя бы один сезон или одного игрока для отображения статистики")
        else:
            # Автопересчет при изменении весов
            # st.button("Рассчитать статистику")
            if True:
                # Обработка нескольких сезонов
                result_df = process_seasons(
                    df_compile_stats,
                    df_history,
                    season_ids,
                    players_input,
                    coef_def,
                    coef_att,
                    metric_weights,
                    division_weights,
                )
                fig1, fig2, fig3, fig4, fig5 = plot_player_ratings(result_df, ",".join(map(str, season_ids)) or "все сезоны")
                with st.expander("📉 Графики по показателям", expanded=False):
                    st.pyplot(fig1)
                    st.pyplot(fig2)
                    st.pyplot(fig3)
                    st.pyplot(fig4)
                    st.pyplot(fig5)
                result_df = rename_columns(result_df)
                st.dataframe(result_df)

    # 3. Визуализация команд
    elif action == "Сезонная статистика команд":
        # Мультивыбор сезонов (или пусто для всех)
        season_ids = st.multiselect("Выберите сезоны", options=available_seasons)

        # Список команд в выбранных сезонах
        df_temp = df_merged.copy()
        if season_ids:
            df_temp = df_temp[df_temp["ID season"].isin(season_ids)]
        teams_in_season = df_temp["ID team"].unique()
        available_teams = sorted(teams_in_season)

        team_ids = st.multiselect("Выберите команды", available_teams, default=available_teams[:4])
        show_roster = st.checkbox("Отобразить состав команд")
        include_goalies = st.checkbox("Учитывать вратарей")

        # Автопересчет при изменении весов
        # st.button("Построить график")
        if True:
            df_players, df_team, \
            fig_total, fig_stacked, fig_scatter, fig_metric, fig_radar = \
                plot_team_ratings(df_compile_stats,
                    df_history,
                    season_ids if season_ids else None,
                    team_ids,
                    coef_def,
                    coef_att,
                    metric_weights,
                                division_weights)
            with st.expander("📉 Графики по показателям", expanded=False):
                st.pyplot(fig_total)
                st.pyplot(fig_stacked)
                st.pyplot(fig_scatter)
                st.pyplot(fig_metric)
                st.pyplot(fig_radar)

            mapping = {'ID team': 'ID команды', 'team_rating': 'рейтинг', 'division': 'дивизион'}
            df_team = df_team.rename(columns=mapping)

            # Если учитывать вратарей — считаем рейтинг в выбранных сезонах/командах и прибавляем
            if include_goalies:
                try:
                    df_goalkeepers = pd.read_csv(r'data/targeted/goalkeepers_data.csv')
                except Exception:
                    df_goalkeepers = pd.read_csv(r'data/raw/goalkeepers_data.csv') if os.path.exists(r'data/raw/goalkeepers_data.csv') else pd.DataFrame()

                allowed_games = set(df_temp['ID game'].dropna().astype(int).unique()) if 'ID game' in df_temp.columns else set()
                team_list = set(team_ids)
                gk_table = compute_goalkeepers_ratings(
                    df_goalkeepers,
                    goalie_metric_weights=saved.get('goalie_metrics', None),
                    allowed_game_ids=allowed_games if allowed_games else None,
                    allowed_team_ids=team_list,
                    include_team_col=True,
                    amplua_weight_gk=coef_gk,
                )
                gk_team_rating = gk_table.groupby('ID team', as_index=False)['рейтинг'].sum().rename(columns={'рейтинг': 'рейтинг_вратарей'})
                df_team = df_team.merge(gk_team_rating, left_on='ID команды', right_on='ID team', how='left').drop(columns=['ID team'])
                df_team['рейтинг_вратарей'] = df_team['рейтинг_вратарей'].fillna(0.0).round(2)
                df_team['рейтинг'] = (df_team['рейтинг'] + df_team['рейтинг_вратарей']).round(2)

            st.dataframe(df_team)

            st.session_state['last_df_players'] = df_players
            st.session_state['last_df_team'] = df_team
            st.session_state['last_team_ids'] = team_ids

            if show_roster:
                st.subheader("Состав команд и рейтинги игроков")
                df_players_all = st.session_state['last_df_players']
                expected_cols = [
                    'ID игрока', 'амплуа', 'игры', 'шайбы', 'о_ш',
                    'ассисты', 'о_а', 'ассисты_2', 'о_а_2', 'броски_мимо', 'о_м',
                    'броски_в_створ', 'о_с', 'блок_броски',
                    'о_б', 'п/м', 'о_п/м', 'общий_рейтинг'
                ]
                for tid in st.session_state['last_team_ids']:
                    st.markdown(f"**Команда ID {tid}**")
                    if include_goalies and 'gk_table' in locals():
                        gk_table_team = gk_table[gk_table['ID team'] == tid]
                        if not gk_table_team.empty:
                            st.markdown("Вратари (выбранные сезоны):")
                            gk_table_team = gk_table_team.rename(columns={'ID team': 'ID команды'})
                            st.dataframe(gk_table_team, use_container_width=True)
                    roster = df_players_all[df_players_all['ID team'] == tid]
                    if roster.empty:
                        st.write("Нет данных по игрокам для этой команды.")
                    else:
                        roster = rename_columns(roster)
                        cols = [c for c in expected_cols if c in roster.columns]
                        st.dataframe(roster[cols], use_container_width=True, height=400,
                                    key=f"roster_{tid}")

    # 4. Рейтинг вратарей
    elif action == "Рейтинг вратарей":
        # Загружаем данные вратарей
        # Загружаем целиком без фильтров — должен быть один ряд на игрока (сумма за всё время)
        try:
            df_goalkeepers = pd.read_csv(r'data/targeted/goalkeepers_data.csv')
        except Exception:
            df_goalkeepers = pd.read_csv(r'data/raw/goalkeepers_data.csv') if os.path.exists(r'data/raw/goalkeepers_data.csv') else pd.DataFrame()

        # Без ограничений по играм и командам и без колонки команды: суммарно за всё время
        gk_table = compute_goalkeepers_ratings(
            df_goalkeepers,
            goalie_metric_weights=goalie_metric_weights,
            allowed_game_ids=None,
            allowed_team_ids=None,
            include_team_col=False,
            amplua_weight_gk=coef_gk,
        )
        st.subheader("Рейтинг вратарей (красный метод)")
        st.dataframe(gk_table, use_container_width=True)