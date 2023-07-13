# библиотеки
import pandas as pd
import schedule as sl
from openpyxl import load_workbook

from google.protobuf.timestamp_pb2 import Timestamp

# стандартные
import sys
import time
from datetime import datetime, timedelta

# модули
import config
import sub
import strategy as st
import analyst as at
from service.tgSend import message_to as msg

sys.path.append('service')
from service import marketdata_pb2, marketdata_pb2_grpc, operations_pb2_grpc, operations_pb2, orders_pb2_grpc, orders_pb2, grpcConnection as gCon
from service.sandbox import money_info


token = config.sandboxToken
account = config.sandbox_id
channel = gCon.conn(config.sandboxApi)


def pick_candles(x, per='1_MIN', j=0):
    try:
        candles_stub = marketdata_pb2_grpc.MarketDataServiceStub(channel)
        t = (datetime.now() - timedelta(days=j + 1)).timestamp()
        seconds1 = int(t)
        nanos1 = int(t % 1 * 1e9)
        t2 = (datetime.now() - timedelta(days=j)).timestamp()
        seconds2 = int(t2)
        nanos2 = int(t2 % 1 * 1e9)
        start_time = Timestamp(seconds=seconds1, nanos=nanos1)
        end_time = Timestamp(seconds=seconds2, nanos=nanos2)
        kwargs = {
            'figi': x.figi,
            'from': start_time,
            'to': end_time,
            'interval': 'CANDLE_INTERVAL_' + per
        }

        historical_candles = candles_stub.GetCandles(marketdata_pb2.GetCandlesRequest(**kwargs), metadata=token)
        return historical_candles.candles
    except Exception as e:
        print(f"Не удалось взять исторические свечи {x['name']}, ОШИБКА:", str(e))


def trading_status(figi):
    try:
        candles_stub = marketdata_pb2_grpc.MarketDataServiceStub(channel)
        ts = candles_stub.GetTradingStatus(marketdata_pb2.GetTradingStatusRequest(figi=figi), metadata=token)
        print(f"статус торгов: {ts.trading_status}")
        return ts.market_order_available_flag
    except Exception as e:
        print(f"Не удалось получить информацию о работе биржи, ОШИБКА:", str(e))


def new_df(candles, x):
    try:
        if not candles:
            return print(f"Пришел пустой набор исторических свеч {x['name']}")
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame([{
            'time': sub.no_timestamp(c.time.seconds),
            'volume': c.volume,
            'open': sub.price(c.open),
            'close': sub.price(c.close),
            'high': sub.price(c.high),
            'low': sub.price(c.low),
            'finish': c.is_complete
        } for c in candles])

        # плохое решение, переделать
        df_p = pd.DataFrame(df['close'])

        # модули аналитики, нужно чистить
        at.vol_density(df)
        at.acum(df, x.smooacum1, x.smooacum2)
        at.donchian(df)
        at.candle(df)
        at.HAiken(df)
        at.arunosc(df)
        at.cci_ind(df, x.per_cci, x.sm_cci)
        at.kauf(df, x.per_kama, x.offset_kama, x.kama_range)
        at.macd(df)
        at.rox(df, x.roc_per, x.sm1roc, x.sm2roc, x.roc_range)
        at.intensiv(df, x.flatTema1, x.smrngT1, x.smrngT2, x.offset1, x.sm1FlatTema1, x.sm2FlatTema1, x.flatRange1)
        at.EMA(df)
        at.pivot(df_p, x.highD, x.lowD, x.closeD, df, x.dev)
        at.priceDensity(df, x.atrPer, x.pdPer, x.sm1pd, x.sm2pd)
        at.angeCalc(df)
        at.bollinger_deal(df)
        at.bollinger_fix(df)
        at.tsindex(df)

        # бесполезно ?
        # df['level'] = df_p['level']
        # df['zona'] = df_p['zona']
        # df['vector_pivot'] = df_p['vector_pivot']

        # подключение стратегий
        st.door(x, df, df_p)
        st.status(x, df, df_p)
        st.fix(x, df, df_p)

        # отслеживание сделок
        mass = []
        for item in operation(x.figi).operations:
            if item.type != 'Удержание комиссии за операцию':
                date = sub.no_timestamp(item.date.seconds)
                mass.append(date)
        df['🌟'] = df.apply(lambda row: sub.check_match(row, mass), axis=1)

        # вывод в консоль
        print(df[['time', 'close', 'door', 'status', '🌟', 'fix']].tail(10))
        return df
    except Exception as e:
        print("Не удалось построить дата фрейм, ОШИБКА:", str(e))


def get_portfolio(figi):  # переделать однажды
    try:
        portfolio_stub = operations_pb2_grpc.OperationsServiceStub(channel)
        info = portfolio_stub.GetPortfolio(operations_pb2.PortfolioRequest(account_id=account), metadata=token)
        found = False
        for item in info.positions:
            if item.figi == figi:
                found = True
                if item.quantity.units < 0:
                    return "inShort"
                if item.quantity.units > 0:
                    return "inLong"
        if not found:
            return "void"
    except Exception as e:
        print("Не удалось получить информацию о портфеле, ОШИБКА:", str(e))


def operation(figi):
    operations_stub = operations_pb2_grpc.OperationsServiceStub(channel)
    t = (datetime.now() - timedelta(days=1)).timestamp()
    seconds1 = int(t)
    nanos1 = int(t % 1 * 1e9)
    t2 = (datetime.now() - timedelta(days=0)).timestamp()
    seconds2 = int(t2)
    nanos2 = int(t2 % 1 * 1e9)
    start_time = Timestamp(seconds=seconds1, nanos=nanos1)
    end_time = Timestamp(seconds=seconds2, nanos=nanos2)
    kwargs = {
        'account_id': account,
        'state': 1,
        'from': start_time,
        'to': end_time,
        'figi': figi
    }
    info = operations_stub.GetOperations(operations_pb2.OperationsRequest(**kwargs), metadata=token)
    return info


def make_deal(status, x):
    if get_portfolio(x.figi) != "void":
        return
    if status == 'BUY':
        go_trade(x['name'], x.figi, x['count'], 'BUY', 'покупка')
    if status == 'SHORT' and x.isShort != 'no':
        go_trade(x['name'], x.figi, x['count'], 'SELL', 'продажа')


def make_fix(x):
    portfolio = get_portfolio(x.figi)
    if portfolio == "void":
        return
    if portfolio == "inShort":
        go_trade(x['name'], x.figi, x['count'], 'BUY', 'фиксация шорта')
    if portfolio == "inLong":
        go_trade(x['name'], x.figi, x['count'], 'SELL', 'фиксация лонга')


def go_trade(name, figi, count, direction, deal):
    try:
        trade_stab = orders_pb2_grpc.OrdersServiceStub(channel)
        kwargs = {
            'figi': figi,
            'quantity': count,
            'direction': f"ORDER_DIRECTION_{direction}",
            'account_id': account,
            'order_type': 'ORDER_TYPE_MARKET',
            'order_id': str(datetime.now().timestamp())
        }
        res = trade_stab.PostOrder(orders_pb2.PostOrderRequest(**kwargs), metadata=token)
        if res.execution_report_status == 5:
            msg(f"заявка исполнена только частично ({name}), исполнено {res.lots_executed} из {res.lots_requested}")
            return print(f"заявка исполнена только частично, исполнено {res.lots_executed} из {res.lots_requested}")
        if res.execution_report_status != 1:
            msg(f"не удалось исполнить заявку {direction} ({name})")
            return print(f"не удалось исполнить заявку {direction}")
        # make_stop()
        money = sub.price(res.total_order_amount)
        commission = sub.price(res.executed_commission.units) if token == config.token else 'недоступна в песочнице'
        msg(f"{deal.upper()} {name.upper()}, {res.lots_executed} ШТ. \nЦена с учетом комиссии {money} \nCумма комиссии {commission}\n{money_info()}")
        print(f"{deal.upper()} {name.upper()} в размере {res.lots_executed} шт. Цена с учетом комиссии {money}, сумма комиссии {commission}")
    except Exception as e:
        print("Не удалось совершить сделку, ОШИБКА:", str(e))


def get_pivot(x, close, high, low):
    workbook = load_workbook('test.xlsx')
    sheet = workbook.active
    sheet['B' + str(x.name + 3)].value = close
    sheet['C' + str(x.name + 3)].value = high
    sheet['D' + str(x.name + 3)].value = low
    workbook.save('test.xlsx')


def preparation():
    excel = sub.load_excel()
    for index, row in excel.iterrows():
        pivot = pick_candles(row, 'DAY')[0]
        get_pivot(row, sub.price(pivot.close), sub.price(pivot.high), sub.price(pivot.low))
    return pd.read_excel('test.xlsx', skiprows=1)


def bot(x):
    if trading_status(x.figi) is not True:
        return print(f"Торги закрыты")
    if not x.startT <= datetime.now().time() <= x.endT:
        return print(f"Торги открыты, но пользовательское время вне диапазона")
    df = new_df(pick_candles(x), x)
    if df.iloc[-1].status == 'BUY' or df.iloc[-1].status == 'SELL':
        make_deal(df.iloc[-1].status, x)
    if df.iloc[-1].fix == 'FIX':
        make_fix(x)


def start_bot():
    for index, row in preparation().iterrows():
        print(f"\n―――――――――――――――――― {row['name'].upper()} ――――――――――――――――――")
        bot(row)
        print('――――――――――――――――――――――――――――――――――――――――')
    print(f"\n■■■ {money_info()} ■■■■■■■■■■■ свечи {(datetime.now().minute - 1)} минуты ■■■■■■■■■■■\n")
    time.sleep(45)


# go_trade('какая то америка', 'BBG00HTN2CQ3', 1829, 'SELL', 'фиксация шорта')


def loop_bot():
    sl.every().minute.at(":50").do(start_bot)
    while True:
        sl.run_pending()


choice = input("'start' чтобы запустить бота, или нажать 'enter' для теста: ")
start_bot() if choice == '' else (loop_bot() if choice == 'start' else print('неверный ввод'))

# loop_bot()
