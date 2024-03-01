import torch
from chess import Board, engine
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Chess(Proxy):
    def __init__(self, engine_path: str, **kwargs):
        """Creates the proxy by giving it the path to the chess engine on the
        local machine."""
        super().__init__(**kwargs)
        self.engine = engine.SimpleEngine.popen_uci(
            engine_path
        )  # the stockfish engine.

    def __call__(self, states: TensorType["batch", 1]) -> TensorType["batch"]:  # type: ignore
        scores = [
            self.engine.analyse(states[i], engine.Limit(time=1))
            for i, _ in enumerate(states)
        ]