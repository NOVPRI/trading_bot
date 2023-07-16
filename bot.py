import pandas as pd  # библиотеки
import schedule as sl
from openpyxl import load_workbook
from google.protobuf.timestamp_pb2 import Timestamp

import time  # стандартные
from datetime import datetime, timedelta

from proto.grpcConnection import conn  # модули
from proto.sandbox import money_info
from proto import marketdata_pb2, orders_pb2, operations_pb2
from service import sub
from analyst import indicators
from service.tgSend import message_to as msg
import strategy as st
import config


def pick_candles(x, per='1_MIN', j=0):
    try:
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
        historical_candles = user.market().GetCandles(marketdata_pb2.GetCandlesRequest(**kwargs), metadata=user.token)
        return historical_candles.candles
    except Exception as e:
        msg(f"Не удалось взять исторические свечи {x['name']}")
        print(f"Не удалось взять исторические свечи {x['name']}, ОШИБКА:", str(e))


def trading_status(figi):
    try:
        ts = user.market().GetTradingStatus(marketdata_pb2.GetTradingStatusRequest(figi=figi), metadata=user.token)
        return ts.market_order_available_flag
    except Exception as e:
        msg("Не удалось получить информацию о работе биржи")
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

        # получение pivot, пока не внедрен
        # pivot = create_pivot(x)

        # подключение индикаторов
        indicators(df, x)

        # подключение стратегий
        st.door(x, df)
        st.status(x, df)
        st.fix(x, df)

        # отслеживание сделок
        mass = []
        for item in operation(x.figi).operations:
            if item.type != 'Удержание комиссии за операцию':
                date = sub.no_timestamp(item.date.seconds)
                mass.append(date)
        df['🌟'] = df.apply(lambda row: sub.check_match(row, mass), axis=1)

        print(df[['time', 'close', 'door', 'status', '🌟', 'fix']].tail(10))
        return df
    except Exception as e:
        msg(f"Не удалось построить дата фрейм {x['name']}")
        print("Не удалось построить дата фрейм, ОШИБКА:", str(e))


def get_portfolio(figi):  # переделать однажды
    try:
        info = user.operation().GetPortfolio(operations_pb2.PortfolioRequest(account_id=user.account), metadata=user.token)
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
        msg(f"Не удалось получить информацию о портфеле")
        print("Не удалось получить информацию о портфеле, ОШИБКА:", str(e))


def operation(figi):
    t = (datetime.now() - timedelta(days=1)).timestamp()
    seconds1 = int(t)
    nanos1 = int(t % 1 * 1e9)
    t2 = (datetime.now() - timedelta(days=0)).timestamp()
    seconds2 = int(t2)
    nanos2 = int(t2 % 1 * 1e9)
    start_time = Timestamp(seconds=seconds1, nanos=nanos1)
    end_time = Timestamp(seconds=seconds2, nanos=nanos2)
    kwargs = {
        'account_id': user.account,
        'state': 1,
        'from': start_time,
        'to': end_time,
        'figi': figi
    }
    info = user.operation().GetOperations(operations_pb2.OperationsRequest(**kwargs), metadata=user.token)
    return info


def make_deal(status, x):
    print('заход в дил')
    if get_portfolio(x.figi) != "void":
        print('в портфеле не пусто')
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
        print('попытка сделки')
        kwargs = {
            'figi': figi,
            'quantity': count,
            'direction': f"ORDER_DIRECTION_{direction}",
            'account_id': user.account,
            'order_type': 'ORDER_TYPE_MARKET',
            'order_id': str(datetime.now().timestamp())
        }
        res = user.order().PostOrder(orders_pb2.PostOrderRequest(**kwargs), metadata=user.token)
        if res.execution_report_status == 5:
            msg(f"заявка исполнена только частично ({name}), исполнено {res.lots_executed} из {res.lots_requested}")
            return print(f"заявка исполнена только частично, исполнено {res.lots_executed} из {res.lots_requested}")
        if res.execution_report_status != 1:
            msg(f"не удалось исполнить заявку {direction} ({name})")
            return print(f"не удалось исполнить заявку {direction}")
        # make_stop()
        money = sub.price(res.total_order_amount)
        commission = sub.price(res.executed_commission.units) if config.sandboxMode else 'недоступна в песочнице'
        msg(f"{deal.upper()} {name.upper()}, {res.lots_executed} ШТ. \nЦена с учетом комиссии {money} \nСумма комиссии {commission}"
            f"\n{money_info() if config.sandboxMode else 'баланс недоступен'}")
        print(f"{deal.upper()} {name.upper()} в размере {res.lots_executed} шт. Цена с учетом комиссии {money}, сумма комиссии {commission}")
    except Exception as e:
        msg(f"Не удалось совершить сделку {name}")
        print("Не удалось совершить сделку, ОШИБКА:", str(e))


def get_pivot(x, close, high, low):
    workbook = load_workbook('service/figi.xlsx')
    sheet = workbook.active
    sheet['B' + str(x.name + 3)].value = close
    sheet['C' + str(x.name + 3)].value = high
    sheet['D' + str(x.name + 3)].value = low
    workbook.save('service/figi.xlsx')


def preparation():
    excel = sub.load_excel()
    for index, row in excel.iterrows():
        pivot = pick_candles(row, 'DAY')[0]
        get_pivot(row, sub.price(pivot.close), sub.price(pivot.high), sub.price(pivot.low))
    return pd.read_excel('service/figi.xlsx', skiprows=1)


def bot(x):
    if trading_status(x.figi) is not True:
        return print(f"Торги закрыты")
    if not x.startT <= datetime.now().time() <= x.endT:
        return print(f"Торги открыты, но пользовательское время вне диапазона")
    df = new_df(pick_candles(x), x)
    if df.iloc[-1].status == 'BUY' or df.iloc[-1].status == 'SELL':
        make_deal(df.iloc[-1].status, x)
    elif df.iloc[-1].fix == 'FIX':
        make_fix(x)


def start_bot():
    for index, row in instruments.iterrows():
        print(f"\n――――――――――――――― {row['name'].upper()} ―{' 🚫 ' if not row.shortly else ''}―――――――――――――――――")
        bot(row)
        print('――――――――――――――――――――――――――――――――――――――――')
    print(f"\n■■■ {money_info() if config.sandboxMode else 'счет недоступен'} ■■■■■■■■■■■ свечи {(datetime.now().minute - 1)} минуты ■■■■■■■■■■■\n")
    time.sleep(45)


def loop_bot():
    sl.every().minute.at(":50").do(start_bot)
    while True:
        sl.run_pending()


if __name__ == "__main__":
    user = conn()
    instruments = None
    if not instruments:
        instruments = preparation()
    choice = input("'start' чтобы запустить бота, или нажать 'enter' для теста: ")
    start_bot() if choice == '' else (loop_bot() if choice == 'start' else print('неверный ввод'))
    # loop_bot()


