def door(x, df):
    if x.st_door == 'bbma':
        df.loc[(df['tema'] > df['BBs_HB']) | (df['tema'] < df['BBs_LB']), 'door'] = 'open'
    else:
        df['door'] = '❌'


def status(x, df):
    if x.st_status == 'first':
        df.loc[(df['emaVector20'] == 'UP') & (df['macd_vector200'] == 'up') & (df['tema'] > df['BBs_ma']) & (df['door'] == 'open') & (
                    df['emaVector200'] == 'UP'), 'status'] = 'BUY'
        df.loc[(df['emaVector20'] == 'DOWN') & (df['macd_vector200'] == 'down') & (df['tema'] < df['BBs_ma']) & (df['door'] == 'open') & (
                    df['emaVector200'] == 'DOWN'), 'status'] = 'SHORT'
    else:
        df['status'] = '❌'


def fix(x, df):
    if x.st_fix == 'B_z_tema':
        df.loc[(df['BBf_HB'].shift() < df['tema'].shift()) & (df['BBf_HB'] > df['tema']), 'fix'] = 'FIX'
        df.loc[(df['BBf_HB'].shift() > df['tema'].shift()) & (df['BBf_HB'] < df['tema']), 'fix'] = '⇈'
        df.loc[(df['BBf_LB'].shift() > df['tema'].shift()) & (df['BBf_LB'] < df['tema']), 'fix'] = 'FIX'
        df.loc[(df['BBf_LB'].shift() < df['tema'].shift()) & (df['BBf_LB'] > df['tema']), 'fix'] = '⇊'
    else:
        df['status'] = '❌'
