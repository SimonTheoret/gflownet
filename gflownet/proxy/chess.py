from chess.engine import SimpleEngine
import torch
from chess import Board, engine
from torch import Tensor
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Chess(Proxy):
    def __init__(self, engine_path: str, **kwargs):
        super().__init__(**kwargs)
        self.engine_path = engine_path
        self.boards = {}

    def setup(self, env=None):
        assert env is not None
        self.boards[id(env)] = env.state

    def __call__(self, states: TensorType["batch", 65]) -> Tensor:
        with SimpleEngine.popen_uci(self.engine_path) as engine:
            scores = [
                engine.analyse(states[i, -1], engine.Limit(time=0.5))
                for i, _ in enumerate(states)
            ]
        return torch.tensor(scores)
