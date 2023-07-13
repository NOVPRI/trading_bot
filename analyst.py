import math
import numpy as num
from ta.momentum import KAMAIndicator
from ta.momentum import ROCIndicator
from ta.momentum import PercentageVolumeOscillator
from ta.momentum import tsi
from ta.trend import CCIIndicator
from ta.trend import aroon_up
from ta.trend import aroon_down
from ta.trend import ema_indicator
from ta.volume import volume_weighted_average_price
from ta.volume import on_balance_volume
from ta.volatility import average_true_range
from ta.volatility import DonchianChannel
from ta.volatility import BollingerBands
from ta.volume import AccDistIndexIndicator
from ta.volume import VolumePriceTrendIndicator
from ta.trend import MACD
from ta.volume import ChaikinMoneyFlowIndicator
from ta.volume import ForceIndexIndicator
from ta.volume import MFIIndicator
import pandas as pd
def tsindex(df):
    df['true_index'] = tsi(close=df['close'], window_fast = 5, window_slow = 100)

#ta.momentum.TSIIndicator
def bollinger_fix (df):
    bolinger = BollingerBands(window=200, window_dev=2.62, close=df['close']) # ШОЙГУ , ГЕРАСИМОВ!!!!КАКОГО ХЕРА Я НЕ МОГУ  ВМЕСТО ДВОЙКИ НАПИСАТЬ 1.9?
    df['BBf_ma'] = bolinger.bollinger_mavg()
    df['BBf_HB'] = bolinger.bollinger_hband()
    df['BBf_LB'] = bolinger.bollinger_lband()
    #df['BBf_width'] = df['BBf_HB'] - df['BBf_LB']
    #df.loc[(df['tema'] > df['BB_HBf']), 'Bollinger_zone_f'] = 'h2'
    #df.loc[(df['tema'] < df['BB_LBf']), 'Bollinger_zone_f'] = 'l2'
    #df.loc[((df['tema'] > df['BB_HBf']) & (df['tema'] < df['BB_HBf'].shift())), 'Bollinger_zone_f'] = 'h2_long'
    #df.loc[((df['tema'] < df['BB_HBf']) & (df['tema'] > df['BB_HBf'].shift())), 'Bollinger_zone_f'] = 'h2_fix'
    #df.loc[(df['tema'] > df['BB_LBf']) & (df['tema'] > df['BB_LBf']), 'Bollinger_zone_f'] = 'l2'

def bollinger_deal (df):
    bolinger = BollingerBands(window=200, window_dev=0.38, close=df['close']) # ШОЙГУ , ГЕРАСИМОВ!!!!КАКОГО ХЕРА Я НЕ МОГУ  ВМЕСТО ДВОЙКИ НАПИСАТЬ 1.9?
    df['BBs_ma'] = bolinger.bollinger_mavg()
    df['BBs_HB'] = bolinger.bollinger_hband()
    df['BBs_LB'] = bolinger.bollinger_lband()
    df['BBs_w'] = bolinger.bollinger_wband()
    #df.loc[(df['tema'] > df['BB_HBs']), 'Bollinger_zone_s'] = 'h04'
    #df.loc[(df['tema'] < df['BB_LBs']), 'Bollinger_zone_s'] = 'l04'
    df['BBs_wp'] = (df['BBs_w'] - df['BBs_w'].shift() / df['BBs_w'].shift())*100
    df.loc[(df['BBs_wp'] > df['BBs_wp'].shift()), 'Bollinger_width'] = 'up'
    df.loc[(df['BBs_wp'] < df['BBs_wp'].shift()), 'Bollinger_width'] = 'down'




def angeCalc(df, range=5, minangle=5):
    angle_ema_per = 5
    df['angleLine'] = ema_indicator(close=df['close'], window=angle_ema_per)
    df['angle'] = abs(num.rad2deg(num.arctan2(df['angleLine'] - df['angleLine'].shift(range), range)))*10
    df.loc[(df['angle'] > minangle), 'angleImpulse'] = 'impulse'


def kauf(df, per_kama, offset_kama, kama_range):
    sma = df['close'].rolling(per_kama).mean()
    wma = df['close'].rolling(per_kama).apply(lambda x: num.sum(num.abs(x - num.mean(x)) * num.arange(1, len(x) + 1)) / num.sum(num.arange(1, len(x) + 1)))
    kama = sma + (wma * 2 / (2 + 1) - wma * 2 / (30 + 1))
    df['kama_line'] = kama
    df.loc[(df['kama_line'] > df['kama_line'].shift()), 'kama_vector'] = 'long'
    df.loc[(df['kama_line'] < df['kama_line'].shift()), 'kama_vector'] = 'short'
    df['change_kama'] = abs(((df['kama_line'] - df['kama_line'].shift(offset_kama)) / df['kama_line'].shift(offset_kama)) * 100)
    df.loc[(df['change_kama'] > kama_range), 'kama_speed'] = 'impulse'
    df.loc[(df['change_kama'] < kama_range), 'kama_speed'] ='flat'

# 'acdist_vector'
def acum(df, smooacum1 , smooacum2):
    acumdist = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    df['acdist'] = acumdist.acc_dist_index()
    #df['ch_acdist'] = abs(((df['acdist'] - df['acdist'].shift())/df['acdist'].shift())*100)
    df['smoothed_acdist'] = round(ema_indicator(ema_indicator(close=df['acdist'], window=smooacum1), window=smooacum2),2)
    df.loc[(df['smoothed_acdist'] < df['smoothed_acdist'].shift()), 'acdist_vector'] = 'distribution'
    df.loc[(df['smoothed_acdist'] > df['smoothed_acdist'].shift()), 'acdist_vector'] = 'accumulation'

def vol_density(df):
    per_vol1 = 3
    per_vol2 = 3
    per_vol3 = 3
    sm_ch1 = 1
    sm_ch2 = 1
    hold = 0.05
    df['vol_dens1'] = volume_weighted_average_price( high= df['high'], low= df['low'], close= df['close'], volume= df['volume'], window= per_vol1)
    df['vol_dens'] = ema_indicator(ema_indicator(close=df['vol_dens1'], window= per_vol2), window= per_vol3)
    df['change_vol_den1'] = abs(((df['vol_dens'] - df['vol_dens'].shift())/df['vol_dens'].shift())*100)
    df['change_vol_den'] = ema_indicator(ema_indicator( close= df['change_vol_den1'], window=sm_ch1),window= sm_ch2)
    df.loc[(df['change_vol_den'] > df['change_vol_den'].shift()), 'change_vol'] = 'up'
    df.loc[(df['change_vol_den'] < df['change_vol_den'].shift()), 'change_vol'] = 'down'
    df.loc[(df['change_vol'] == 'up') & (df['change_vol_den'] > hold), 'vector_change_vol_den'] = 'strong_volume'
    df.loc[(df['change_vol'] == 'down') & (df['change_vol_den'] > hold), 'vector_change_vol_den'] = 'Pivot'
    df.loc[(df['vol_dens'] > df['vol_dens'].shift()), 'vector_vol_dens'] = 'volume_impulse'
    df.loc[(df['vol_dens'] < df['vol_dens'].shift()), 'vector_vol_dens'] = 'volume_flat'
    df.loc[(df['vector_vol_dens'] == 'volume_flat') & (df['vector_vol_dens'].shift() == 'volume_impulse'), 'vector_volume'] = 'exit'
    df.loc[(df['vector_vol_dens'] == 'volume_impulse') & (df['vector_vol_dens'].shift() == 'volume_flat'), 'vector_volume'] = 'enter'
    #'vector_change_vol_den', 'vector_vol_dens', 'vpt_vector', 'pvo_vector',
# .....................VOLATILITY

def donchian(df): #per_don
    per_don_emka = 20
    per_don_emka2 = 5
    don = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=20, offset=0)
    df['don_up'] = don.donchian_channel_hband()
    df['don_down'] = don.donchian_channel_lband()
    df['don_average'] = don.donchian_channel_mband()
    df['don_width'] = don.donchian_channel_wband()
    df['don_avs'] = ema_indicator(close=df['don_average'], window=5)
    df['emka'] = ema_indicator(ema_indicator(close= df['close'], window= per_don_emka), window= per_don_emka2)
    df.loc[(df['don_down'] == df['don_down'].shift()),  'don_level'] = 'support'
    df.loc[(df['don_up'] == df['don_up'].shift()), 'don_level'] = 'resistance'
    df['don_dist'] = abs(df['don_average'] - df['emka']) # расстояние до средней линии должно сокращаться  а также должно быть либо сопротивление либо поддержка
    #df.loc[(df['don_average'] > df['don_average'].shift()) , 'don_vector'] = 'positive'
    #df.loc[(df['don_average'] < df['don_average'].shift()) , 'don_vector'] = 'negative'
    #df.loc[(df['don_average'] == df['don_average'].shift()), 'don_vector'] = 'sleep'
    df.loc[(df['don_down'] =='support') & (df['emka'] > df['emka'].shift()), 'don_vector'] = 'positive'
    df.loc[(df['don_up'] == 'resistance') & (df['emka'] < df['emka'].shift()), 'don_vector'] = 'negative'
    # 'don_vector' 'don_dist'

def candle(df):
    cdl= (df['high'] + df['low'])/1.99
    df.loc[(df['close'] > cdl), 'candles'] = 'bull_hammer'# тоже что пин-бар ( Пинокио) закрытие выше серердины спреда свечи
    df.loc[(df['close'] < cdl), 'candles'] = 'bear_hammer'  # тоже что пин-бар ( Пинокио) закрытие выше серердины спреда свечи
    df.loc[(df['open'] > df['close'].shift()) & (df['close'] > df['open'].shift()), 'candles'] = 'bear_absorbing' # поглощение
    df.loc[(df['open'] < df['close'].shift()) & (df['close'] < df['open'].shift()), 'candles'] = 'bull_absorbing' # поглощение
    df.loc[(df['open'] == df['close'].shift()) & (df['close'] == df['open'].shift()), 'candles'] = 'rails'


def HAiken(df):
    df['HA_Close'] = round(((df['open'] + df['high'] + df['low'] + df['close']) / 4), 3)
    ha_close_values = df['HA_Close'].values
    length = len(df)
    ha_open = num.zeros(length, dtype=float)
    ha_open[0] = (df['open'][0] + df['close'][0]) / 2
    for i in range(0, length - 1):
        ha_open[i + 1] = round(((ha_open[i] + ha_close_values[i]) / 2), 3)
    df['HA_Open'] = ha_open
    df['HA_High'] = round(df[['HA_Open', 'HA_Close', 'high']].max(axis=1), 3)
    df['HA_Low'] = round(df[['HA_Open', 'HA_Close', 'low']].min(axis=1), 3)
    df['HA_EMA'] = df['HA_Close']
    df.loc[(df['HA_Open'] > df['HA_Close']), 'HA_vector'] = 'down'
    df.loc[(df['HA_Open'] < df['HA_Close']), 'HA_vector'] = 'up'


def arunosc(df):
    sm_arun_osc = 5
    length = 10
    high = df['high']  # предполагаем, что данные хранятся в DataFrame df
    low = df['low']
    highestbars = high.rolling(window=length+1).max().shift(1)  # сдвигаем на 1 бар назад
    upper = 100 * (highestbars + length) / length
    lowestbars = low.rolling(window=length+1).min().shift(1)
    lower = 100 * (lowestbars + length) / length
    df['osc_arun'] = upper - lower
    df['arun_osc'] = ema_indicator(ema_indicator(ema_indicator(close=df['osc_arun'], window= sm_arun_osc),window= sm_arun_osc), window= sm_arun_osc)
    df.loc[(df['arun_osc'] > df['arun_osc'].shift()), 'arun_vector'] = 'up'
    df.loc[(df['arun_osc'] < df['arun_osc'].shift()), 'arun_vector'] = 'down'

# 'cci_vector'
def cci_ind(df, per_cci, sm_cci):
    cci_indicator = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=per_cci, constant=0.015)
    df['cci'] = cci_indicator.cci()
    df['sm_cci'] = ema_indicator(ema_indicator(close=df['cci'], window=sm_cci), window= sm_cci)
    df.loc[(df['sm_cci'] >= df['sm_cci'].shift()) , 'cci_vector'] = 'long'
    df.loc[(df['sm_cci'] < df['sm_cci'].shift()) , 'cci_vector'] = 'short'

# trend/ anti_trend =====================================================
# 'kama_vector'



def macd(df): #'macd', 'macd_signal'
    slow_per = 200
    fast_per = 10
    per_smacd = 3
    slow_per2 = 34
    fast_per2 = 10
    per_smacd2 = 3
    ema_macd1 = ema_indicator(close=df['close'], window=fast_per)
    ema_macd2 = ema_indicator(close=df['close'], window=slow_per)
    df['macds1'] = ((ema_macd1 - ema_macd2)/ema_macd2)*100
    df['macd1'] = ema_indicator(close=df['macds1'], window=per_smacd) #сглаженное процентное расхождение
    ema_macd3 = ema_indicator(close=df['close'], window=fast_per2)
    ema_macd4 = ema_indicator(close=df['close'], window=slow_per2)
    df['macds2'] = ((ema_macd3 - ema_macd4)/ema_macd4)*100
    df['macd2'] = ema_indicator(close=df['macds2'], window=per_smacd2)  # сглаженное процентное расхождение

    df.loc[(df['macd1'] < 0.05) & (df['macd1'] > -0.05), 'macd_zone'] = 'Стагнирует'
    df.loc[(df['macd1'] < 0.75) & (df['macd1'] > 0.05), 'macd_zone'] = 'Движуха'
    df.loc[(df['macd1'] > -0.75) & (df['macd1'] < -0.05), 'macd_zone'] = 'Движуха'
    df.loc[(df['macd1'] > df['macd2']), 'macd_vector'] = 'down'
    df.loc[(df['macd1'] > 0.75), 'macd_zone'] = 'Перекуп'
    df.loc[(df['macd1'] > 0.88), 'macd_zone'] = 'Сильный перекуп'
    df.loc[(df['macd1'] < df['macd2']), 'macd_vector'] = 'up'
    df.loc[(df['macd1'] < -0.75), 'macd_zone'] = 'Перепрод'
    df.loc[(df['macd1'] < -0.88), 'macd_zone'] = 'Сильный перепрод'

'''
def emadist(df, perFix1, perFix2, smFix1, smFix2, dist_range):
    df['distema1'] = ema_indicator(ema_indicator(close=df['close'], window=perFix1), window=smFix1)# 3 3
    df['distema2'] = ema_indicator(ema_indicator(close=df['close'], window=perFix2), window=smFix2)# 20 5
    df['vectordist'] = abs((df['distema1'] - df['distema2']) * 10)
    # тут тупость
    # логика такова что если расстояние между 1й и 2й скользящими увеличивается значит ТРЕНД (тоесть 1я отдаляется от 2й)
    df.loc[(df['vectordist'] > df['vectordist'].shift()), 'distvector'] = 'TREND'#  & (df['vectordist'] > dist_range) РАССТОЯНИЕ УВЕЛИЧИВАЕТСЯ
    # df.loc[(df['vectordist'] < df['vectordist'].shift()), 'distvector'] = 'CNTR_TREND' = NaN?
'''
# SIDEWAYS_detection
# 'roc_vector'
def rox(df, roc_per, sm1roc, sm2roc, roc_range): #ЭТО ТОЖЕ САМОЕ ЧТО И РАЗМАХ( В СТАТИСТИКЕ) ... ИЗМЕРЯЕТ АМЛИТУДУ за определенный период
    roxi = ROCIndicator(close=df['close'], window=roc_per)
    df['roc'] = roxi.roc()
    df['vector_roc'] = ema_indicator(ema_indicator(close=df['roc'], window=sm1roc), window=sm2roc)
    df.loc[(df['vector_roc'] < df['vector_roc'].shift()) & (df['vector_roc'] < roc_range), 'roc_vector'] = 'DOWN'
    df.loc[(df['vector_roc'] >= df['vector_roc'].shift()) & (df['vector_roc'] > roc_range), 'roc_vector'] = 'UP'




def priceDensity(df, atrPer, pdPer, sm1pd, sm2pd):
    # === Если плотность цены растет значит цена зажимается в более узком диапазоне и это флэт.
    # === Если цена наоборот кудато летит то плотность будет падать и это импульс
    df['atr'] = average_true_range(high=df['high'], low=df['low'], close=df['close'], window=atrPer)
    df['chislitel'] = df['atr'].rolling(pdPer).sum()
    df['znamenatel'] = df['high'].rolling(pdPer).max() - df['low'].rolling(pdPer).min()
    df['pd'] = df['chislitel'] / df['znamenatel']
    df['smoothedPD'] = round(ema_indicator(close=ema_indicator(close=df['pd'], window=sm1pd), window=sm2pd), 2)
    df.loc[(df['smoothedPD'] >= df['smoothedPD'].shift()), 'priceDensity'] = 'flat'
    df.loc[(df['smoothedPD'] < df['smoothedPD'].shift()), 'priceDensity'] = 'impulse'


def intensiv(df, flatTema1, smrngT1, smrngT2, offset1, sm1FlatTema1, sm2FlatTema1, flatRange1):
    # =================ИНТЕНСИВНОСЬ ИЗМЕНЕНИЯ СКОЛЬЗЯЩЕЙ ====================================================
    df['Tema13'] = ema_indicator(close=df['close'], window=flatTema1)
    df['Tema23'] = ema_indicator(close=df['Tema13'], window=flatTema1)
    df['Tema33'] = ema_indicator(close=df['Tema23'], window=flatTema1)
    df['temaF3'] = 3 * (df['Tema13'] - df['Tema23']) + df['Tema33']

    df['flatTema'] = ema_indicator(close=ema_indicator(close=df['temaF3'], window=smrngT1), window=smrngT2)
    df['flat'] = abs(((df['flatTema'] - df['flatTema'].shift(offset1)) / df['flatTema'].shift(offset1)) * 100)
    df.loc[(df['flatTema'] > df['flatTema'].shift()), 'isFlat1'] = 'up'
    df.loc[(df['flatTema'] < df['flatTema'].shift()), 'isFlat1'] = 'down'
    df.loc[(df['flat'] > flatRange1) & (df['isFlat1'] == 'up'), 'tema_speed'] = 'long'
    df.loc[(df['flat'] > flatRange1) & (df['isFlat1'] == 'down'), 'tema_speed'] = 'short'

def EMA(df):
    df['HA_Close'] = round(((df['open'] + df['high'] + df['low'] + df['close']) / 4), 3)
    df['ema71'] = ema_indicator(close=df['close'], window=3)
    df['ema72'] = ema_indicator(close=df['ema71'], window=3)
    df['ema73'] = ema_indicator(close=df['HA_Close'],window=20)
    df.loc[(df['ema72'] > df['ema72'].shift()), 'emaVector'] = 'UP'
    df.loc[(df['ema72'] < df['ema72'].shift()), 'emaVector'] = 'DOWN'
    df.loc[(df['emaVector'] != df['emaVector'].shift()), 'signal_ema'] = 'Переворот'

# PIVOT
def tema(df):
    df['Tema13'] = ema_indicator(close=df['close'], window=1)

    df['Tema23'] = ema_indicator(close=df['Tema13'], window=1)
    df['Tema33'] = ema_indicator(close=df['Tema23'], window=1)
    df['temaF3'] = 3 * (df['Tema13'] - df['Tema23']) + df['Tema33']
    df['tema'] = ema_indicator(close=ema_indicator(close=df['temaF3'], window=1), window=1)

def supertrend(df, atr_period, multiplier):
    high = df['high']
    low = df['low']
    close = df['close']

    # calculate ATR
    price_diffs = [high - low,
                   high - close.shift(),
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()
    # df['atr'] = df['tr'].rolling(atr_period).mean()

    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)

    # initialize Supertrend column to True
    supertrend = [True] * len(df)

    for i in range(1, len(df.index)):
        curr, prev = i, i - 1

        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]

            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = num.nan
        else:
            final_lowerband[curr] = num.nan

    return pd.DataFrame({
        'st': supertrend,
        'flow': final_lowerband,
        'fup': final_upperband
    }, index=df.index)



# def LRC(df):
#     period = 20
#     # Рассчитываем коэффициенты регрессии
#     x = num.arange(period)
#     y = data[-period:]
#     coefficients = num.polyfit(x, y, 1)
#     slope, intercept = coefficients[0], coefficients[1]
#
#     # Рассчитываем верхнюю и нижнюю границы канала
#     upper_channel = slope * x[-1] + intercept + 2 * num.std(y)
#     lower_channel = slope * x[-1] + intercept - 2 * num.std(y)
#
#     return slope, intercept, upper_channel, lower_channel


def pivot(df_p, h, l, c,df, dev):
    df_p['close'] = df['close']
    df_p['Tema13'] = ema_indicator(close=df_p['close'], window=1)
    df_p['Tema23'] = ema_indicator(close=df_p['Tema13'], window=1)
    df_p['Tema33'] = ema_indicator(close=df_p['Tema23'], window=1)
    df_p['temaF3'] = 3 * (df_p['Tema13'] - df_p['Tema23']) + df_p['Tema33']
    df_p['tema'] = ema_indicator(close=ema_indicator(close=df_p['temaF3'], window=1), window=1)
    df_p['Tema5'] =ema_indicator(close=df_p['close'], window=5)
    df_p['highD'] = h
    df_p['lowD'] = l
    df_p['closeD'] = c
    #df_p['P'] , df_p['s1'], df_p['r1'], df_p['s2'], df_p['r2'], df_p['s3'] , df_p['r3'] , df_p['s4'] , df_p['r4'] , df_p['s5'] , df_p['r5']

    df_p['P'] = (df_p['highD'] + df_p['lowD'] + df_p['closeD'])/3
    df_p['s1'] = 2 * df_p['P'] - df_p['highD']
    df_p['r1'] = 2 * df_p['P'] - df_p['lowD']
    df_p['s2'] = df_p['P'] - (df_p['highD'] - df_p['lowD'])
    df_p['r2'] = df_p['P'] + (df_p['highD'] - df_p['lowD'])
    df_p['s3'] = df_p['P'] * 2 - (2*df_p['highD'] - df_p['lowD'])
    df_p['r3'] = df_p['P'] * 2 + (df_p['highD'] - 2*df_p['lowD'])
    df_p['s4'] = df_p['P'] * 3 - (3*df_p['highD'] - df_p['lowD'])
    df_p['r4'] = df_p['P'] * 3 + (df_p['highD'] - 3*df_p['lowD'])
    df_p['s5'] = df_p['P'] * 4 - (4*df_p['highD'] - df_p['lowD'])
    df_p['r5'] = df_p['P'] * 4 + (df_p['highD'] - 4*df_p['lowD'])



    #/////////////////////////////////////
    df_p['r05'] = (df_p['r1'] + df_p['P'])/2
    df_p['s05'] = (df_p['s1'] + df_p['P'])/2
    df_p['r15'] = (df_p['r2'] + df_p['r1']) / 2
    df_p['s15'] = (df_p['s1'] + df_p['s2']) / 2
    df_p['r25'] = (df_p['r2'] + df_p['r3']) / 2
    df_p['s25'] = (df_p['s2'] + df_p['s3']) / 2
    df_p['r35'] = (df_p['r3'] + df_p['r4']) / 2
    df_p['s35'] = (df_p['s3'] + df_p['s4']) / 2
    df_p['r45'] = (df_p['r4'] + df_p['r5']) / 2
    df_p['s45'] = (df_p['s4'] + df_p['s5']) / 2
    #/////////////////////////////////////////



    #/////////////////
    df_p['P_upper_band'] = df_p['P'] + df_p['P'] * dev / 100
    df_p['P_lower_band'] = df_p['P'] - df_p['P'] * dev / 100
    df_p['s05_upper_band'] = df_p['s05'] + df_p['s05'] * dev / 100
    df_p['s05_lower_band'] = df_p['s05'] - df_p['s05'] * dev / 100
    df_p['s1_upper_band'] = df_p['s1'] + df_p['s1'] * dev / 100
    df_p['s1_lower_band'] = df_p['s1'] - df_p['s1'] * dev / 100
    df_p['s15_upper_band'] = df_p['s15'] + df_p['s15'] * dev / 100
    df_p['s15_lower_band'] = df_p['s15'] - df_p['s15'] * dev / 100
    df_p['s2_upper_band'] = df_p['s2'] + df_p['s2'] * dev / 100
    df_p['s2_lower_band'] = df_p['s2'] - df_p['s2'] * dev / 100
    df_p['s25_upper_band'] = df_p['s25'] + df_p['s25'] * dev / 100
    df_p['s25_lower_band'] = df_p['s25'] - df_p['s25'] * dev / 100
    df_p['s3_upper_band'] = df_p['s3'] + df_p['s3'] * dev / 100
    df_p['s3_lower_band'] = df_p['s3'] - df_p['s3'] * dev / 100
    df_p['s35_upper_band'] = df_p['s35'] + df_p['s35'] * dev / 100
    df_p['s35_lower_band'] = df_p['s35'] - df_p['s35'] * dev / 100
    df_p['s4_upper_band'] = df_p['s4'] + df_p['s4'] * dev / 100
    df_p['s4_lower_band'] = df_p['s4'] - df_p['s4'] * dev / 100
    df_p['s45_upper_band'] = df_p['s45'] + df_p['s45'] * dev / 100
    df_p['s45_lower_band'] = df_p['s45'] - df_p['s45'] * dev / 100
    df_p['s5_upper_band'] = df_p['s5'] + df_p['s5'] * dev / 100
    df_p['s5_lower_band'] = df_p['s5'] - df_p['s5'] * dev / 100
    df_p['r05_upper_band'] = df_p['r05'] + df_p['r05'] * dev / 100
    df_p['r05_lower_band'] = df_p['r05'] - df_p['r05'] * dev / 100
    df_p['r1_upper_band'] = df_p['r1'] + df_p['r1'] * dev / 100
    df_p['r1_lower_band'] = df_p['r1'] - df_p['r1'] * dev / 100
    df_p['r15_upper_band'] = df_p['r15'] + df_p['r15'] * dev / 100
    df_p['r15_lower_band'] = df_p['r15'] - df_p['r15'] * dev / 100
    df_p['r2_upper_band'] = df_p['r2'] + df_p['r2'] * dev / 100
    df_p['r2_lower_band'] = df_p['r2'] - df_p['r2'] * dev / 100
    df_p['r25_upper_band'] = df_p['r25'] + df_p['r25'] * dev / 100
    df_p['r25_lower_band'] = df_p['r25'] - df_p['r25'] * dev / 100
    df_p['r3_upper_band'] = df_p['r3'] + df_p['r3'] * dev / 100
    df_p['r3_lower_band'] = df_p['r3'] - df_p['r3'] * dev / 100
    df_p['r35_upper_band'] = df_p['r35'] + df_p['r35'] * dev / 100
    df_p['r35_lower_band'] = df_p['r35'] - df_p['r35'] * dev / 100
    df_p['r4_upper_band'] = df_p['r4'] + df_p['r4'] * dev / 100
    df_p['r4_lower_band'] = df_p['r4'] - df_p['r4'] * dev / 100
    df_p['r45_upper_band'] = df_p['r45'] + df_p['r45'] * dev / 100
    df_p['r45_lower_band'] = df_p['r45'] - df_p['r45'] * dev / 100
    df_p['r5_upper_band'] = df_p['r5'] + df_p['r5'] * dev / 100
    df_p['r5_lower_band'] = df_p['r5'] - df_p['r5'] * dev / 100

    df_p.loc[(df_p['P_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['P_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['P_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['P_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['P_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s05_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s05_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s05_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s05_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s05_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s05_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s1_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s1_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s1_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s1_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s1_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s1_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s15_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s15_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s15_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s15_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s15_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s15_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s2_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s2_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s2_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s2_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s2_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s2_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s25_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s25_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s25_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s25_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s25_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s25_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s3_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s3_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s3_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s3_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s3_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s3_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s35_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s35_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s35_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s35_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s35_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s35_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s4_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s4_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s4_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s4_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s4_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s4_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s45_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s45_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s45_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s45_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s45_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s45_lower_band'] > df_p['close']) & (df_p['close'] > df_p['s5_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s5_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['s5_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['s5_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['s5_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['s5_lower_band'] > df_p['close']), 'zona'] = 'over_s5'
    ###///////////////////////////////////////////////////////////////////////////////////////////////
    df_p.loc[(df_p['r05_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r05_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r05_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r05_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r05_lower_band'] > df_p['close']) & (df_p['close'] > df_p['P_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r1_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r1_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r1_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r1_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r1_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r05_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r15_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r15_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r15_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r15_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r15_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r1_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r2_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r2_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r2_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r2_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r2_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r15_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r25_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r25_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r25_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r25_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r25_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r2_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r3_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r3_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r3_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r3_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r3_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r25_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r35_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r35_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r35_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r35_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r35_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r3_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r4_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r4_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r4_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r4_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r4_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r35_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r45_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r45_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r45_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r45_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r45_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r4_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['r5_upper_band'] > df_p['tema'].shift()) & (df_p['tema'] > df_p['r5_upper_band']), 'level'] = 'long'
    df_p.loc[(df_p['r5_lower_band'] < df_p['tema'].shift()) & (df_p['tema'] < df_p['r5_lower_band']), 'level'] = 'short'
    df_p.loc[(df_p['r5_lower_band'] > df_p['close']) & (df_p['close'] > df_p['r45_upper_band']), 'zona'] = 'open'
    df_p.loc[(df_p['s5'] < df_p['Tema5']) & (df_p['Tema5'].shift() < df_p['Tema5']), 'vector_pivot'] = 'long'
    df_p.loc[(df_p['r5'] > df_p['Tema5']) & (df_p['Tema5'].shift() > df_p['Tema5']), 'vector_pivot'] = 'short'
