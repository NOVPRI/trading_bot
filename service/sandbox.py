import config
import sandbox_pb2
import sandbox_pb2_grpc
import common_pb2
import sub
from service import operations_pb2_grpc, operations_pb2, grpcConnection as gCon

token = config.sandboxToken
channel = gCon.conn(config.sandboxApi)
account = config.sandbox_id


def new_account():
    account_stub = sandbox_pb2_grpc.SandboxServiceStub(channel)
    acc = account_stub.OpenSandboxAccount(sandbox_pb2.OpenSandboxAccountRequest(), metadata=token)
    print(acc)


def add_money():
    account_stub = sandbox_pb2_grpc.SandboxServiceStub(channel)
    price = 500000.0
    kwargs = {
        'currency': 'RUB',
        'units': sub.nano_price(price, 'u'),
        'nano': sub.nano_price(price, 'n'),
    }
    value = common_pb2.MoneyValue(**kwargs)
    money = account_stub.SandboxPayIn(sandbox_pb2.SandboxPayInRequest(account_id=account, amount=value),
                                      metadata=token)
    print(money)


def money_info():
    try:
        account_stub = sandbox_pb2_grpc.SandboxServiceStub(channel)
        money = account_stub.GetSandboxWithdrawLimits(sandbox_pb2.SandboxPayInRequest(account_id=account), metadata=token)
        balance = ""
        for item in money.money:
            balance += f"{item.currency} = {sub.price(item)} | "
        return f'Баланс: {balance.rstrip("| ")}'
    except Exception as e:
        print("Не удалось узнать баланс, ОШИБКА:", str(e))


def get_portfolio():
    portfolio_stub = operations_pb2_grpc.OperationsServiceStub(channel)
    info = portfolio_stub.GetPortfolio(operations_pb2.PortfolioRequest(account_id=account), metadata=token)
    print("\nсодержимое портфеля:")
    for item in info.positions:
        if item.figi != 'RUB000UTSTOM' and item.figi != 'BBG0013HGFT4':
            print(str(item.figi)+" "+str(item.quantity))


if __name__ == "__main__":
    # new_account() # создать новый аккаунт
    # add_money() # добавить денег
    print(money_info())
    get_portfolio()

