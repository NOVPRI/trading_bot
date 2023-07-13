import analyst as at


def door(x, df, df_p):
    if x.st_door == 'bbma':
        df.loc[(df_p['tema'] > df['BBs_HB']) | (df_p['tema'] < df['BBs_LB']), 'door'] = 'open'
    else:
        df['door'] = '❌'


def status(x, df, df_p):
    if x.st_status == 'first':
        df.loc[(df['emaVector'] == 'UP') & (df['macd_vector'] == 'up') & (df_p['tema'] > df['BBs_ma']) & (df['door'] == 'open'), 'status'] = 'BUY'
        df.loc[(df['emaVector'] == 'DOWN') & (df['macd_vector'] == 'down') & (df_p['tema'] < df['BBs_ma']) & (df['door'] == 'open'), 'status'] = 'SELL'
    else:
        df['status'] = '❌'


def fix(x, df, df_p):
    if x.st_fix == 'B_z_tema':
        df.loc[(df['BBf_HB'].shift() < df_p['tema'].shift()) & (df['BBf_HB'] > df_p['tema']), 'fix'] = 'FIX'
        df.loc[(df['BBf_LB'].shift() > df_p['tema'].shift()) & (df['BBf_LB'] < df_p['tema']), 'fix'] = 'FIX'
    else:
        df['status'] = '❌'
