import grpc


def conn(api):
    credentials = grpc.ssl_channel_credentials(root_certificates=None, private_key=None, certificate_chain=None)
    channel = grpc.secure_channel(target=api, credentials=credentials)
    return channel
