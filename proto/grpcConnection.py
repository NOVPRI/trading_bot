import grpc
import config
import sys
module_path = sys.path.append('proto')
from proto import marketdata_pb2, orders_pb2_grpc, operations_pb2_grpc, orders_pb2, marketdata_pb2_grpc, operations_pb2


class Connection:
    def __init__(self, token, account, api):
        self.token = token
        self.account = account
        self.api = api
        self.channel = self.__channel()

    def __channel(self):
        credentials = grpc.ssl_channel_credentials(root_certificates=None, private_key=None, certificate_chain=None)
        return grpc.secure_channel(target=self.api, credentials=credentials)

    def market(self):
        return marketdata_pb2_grpc.MarketDataServiceStub(self.channel)

    def operation(self):
        return operations_pb2_grpc.OperationsServiceStub(self.channel)

    def order(self):
        return orders_pb2_grpc.OrdersServiceStub(self.channel)


def conn():
    if config.sandboxMode:
        return Connection(config.sandboxToken, config.sandboxAccountId, config.sandboxApi)
    else:
        return Connection(config.releaseToken, config.releaseAccountId, config.releaseApi)



