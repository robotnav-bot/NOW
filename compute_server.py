import dataclasses
import enum
import logging
import socket


from server import websocket_policy_server
from sim_test.NavPolicy import NavPolicy

def main() -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        host="0.0.0.0",
        port=80,
        metadata=None,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()