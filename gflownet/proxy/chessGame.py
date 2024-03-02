import torch
from chess import Board, engine
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class Chess(Proxy):
    def __init__(self, engine_path: str, board_check: bool = False, **kwargs):
        """Creates the proxy by giving it the path to the chess engine on the
        local machine."""
        super().__init__(**kwargs)
        self.board_check = board_check
        self.engine = engine.SimpleEngine.popen_uci(
            engine_path
        )  # the stockfish engine.
        self.boards = {}  # list of GFlowChess environments. Populated with self.setup()

    def setup(self, env=None):
        """Overrides the default setup function from the base class."""
        assert env is not None
        self.boards[env.id] = env.board

    def __call__(self, states: TensorType["batch", 64]) -> TensorType["batch"]:  # type: ignore
        analyses = [
            self.engine.analyse(self.boards[key], engine.Limit(time=1,depth=20))
            for key in self.boards.keys()
        ]
        scores = []
        for analyse in analyses :
            scores.append(analyse['score'].relative.score())
        return tfloat(scores, device=self.device, float_type=self.float)