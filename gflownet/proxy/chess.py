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
        self.boards = {}  # list of GFlowChess environments. Populated with self.setup()

    def setup(self, env=None):
        """Overrides the default setup function from the base class."""
        assert env is not None
        self.boards[id(env)] = env.state  # ids are string

    def __call__(self, states: TensorType["batch", 65]) -> TensorType["batch"]:  # type: ignore
        scores = [
            self.engine.analyse(states[i, -1], engine.Limit(time=0.5))  # type: ignore
            for i, _ in enumerate(states)
        ]
        return torch.tensor(scores)  # type: ignore
