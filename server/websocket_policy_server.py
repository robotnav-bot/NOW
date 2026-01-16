import asyncio
import http
import logging
import time
import traceback

from server import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from PIL import Image
from sim_test.NavPolicy import NavPolicy
import numpy as np

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        # policy,
        # model_type,
        host: str = "0.0.0.0",
        port: int = None,
        metadata: dict = None,
    ) -> None:
        self._policy = None
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        
        logging.getLogger("websockets.server").setLevel(logging.INFO)




    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()

                result={
                    "action": np.zeros(2),  
                    "server_ready": False,
                    "policy_ready": False,
                }

                if obs["start_new_test"]:
                    model_type=obs["model_type"]
                    goal_type=obs["goal_type"]
                    scene_name=obs["scene_name"]

                    exp_eval = None
                    model_path = None
                    plan_init_type = "anchor"
                    image_text_model_type = "siglip"

                    if model_type == "ours":
                        exp_eval = "config/config_shortcut_w_pretrain.yaml"
                        model_path = f"./logs/{scene_name}_w_pretrain/checkpoints/0050000.pth.tar"
                 
                    self._policy = NavPolicy(
                        model_type=model_type,
                        goal_type=goal_type,
                        task_path=f"./tasks/tasks-{scene_name}-{goal_type}",
                        exp_eval = exp_eval,
                        model_path = model_path,
                        dist_model_path="/root/workspace/code/Navigation_worldmodel/weights/EffoNAV/Effo_nav.pth",
                        plan_init_type=plan_init_type,
                        image_text_model_type=image_text_model_type
                        )
                        
                    result["policy_ready"]=True
                    logger.info("policy ready")
                else:
                    if self._policy is not None:
                        new_result = self._policy.model_compute_v_w(obs)
                        result.update(new_result)

                infer_time = time.monotonic() - infer_time

                result["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None