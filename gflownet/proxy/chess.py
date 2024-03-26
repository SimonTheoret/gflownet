from chess.engine import SimpleEngine
import torch
from chess import Board, engine
from torch import Tensor
from torchtyping import TensorType
from typing import Callable
from functools import partial
import concurrent.futures

from gflownet.proxy.base import Proxy


def default_proxy_score(centipawn: int|None) ->  float:
    if centipawn is not None:
        return 1 / (1 + 10 ** (centipawn/4))
    else:
        return 0.0

class Chess(Proxy):
    def __init__(self, engine_path: str, **kwargs):
        super().__init__(**kwargs)
        self.engine_path = engine_path
        self.boards = {}
        self.default_scorer = partial(self.compute_single_score, time_limit = 0.5, proxy_score = default_proxy_score)

    def setup(self, env=None):
        assert env is not None
        self.boards[id(env)] = env.state

    def __call__(self, states: TensorType["batch", 64]) -> Tensor:
        copied = states.copy()
        copied = copied.tolist()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.default_scorer, copied)
        return torch.tensor(copied)

    def compute_single_score(self, state: Board, time_limit: float, proxy_score: Callable[[int|None],float]) -> float:
        with SimpleEngine.popen_uci(self.engine_path) as eng:
            centipawn = eng.analyse(state, engine.Limit(time=time_limit), info=engine.INFO_SCORE)["score"].relative.score()
        return proxy_score(centipawn)



