COLUMN_MAPPING = {
    'ID player': 'ID игрока', 'amplua': 'амплуа', 'games': 'игры',
    'goals': 'шайбы', 'p_goals': 'о_ш', 'assists': 'ассисты',
    'p_assists': 'о_а', 'throws_by': 'броски_мимо', 'p_throws_by': 'о_м',
    'shot_on_target': 'броски_в_створ', 'p_shot_on_target': 'о_с',
    'blocked_throws': 'блок_броски', 'p_blocked_throws': 'о_б',
    'p_m': 'п/м', 'player_rating': 'общий_рейтинг'
}

def rename_columns(df):
    return df.rename(columns=COLUMN_MAPPING)