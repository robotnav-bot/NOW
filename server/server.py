import dataclasses
import enum
import logging
import socket

import tyro

import websocket_policy_server

def main() -> None:
    # policy = create_policy(args)
    # policy_metadata = policy.metadata

    # # Record the policy's behavior.
    # if args.record:
    #     policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=None,
        host="0.0.0.0",
        port=80,
        metadata=None,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()