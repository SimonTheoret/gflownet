from chess.engine import SimpleEngine
import torch
from chess import Board, engine
from torch import Tensor
from torchtyping import TensorType
from typing import Callable
from functools import partial
import concurrent.futures

from gflownet.proxy.base import Proxy



class Chess(Proxy):
    def __init__(self, engine_path: str, **kwargs):
        super().__init__(**kwargs)
        self.engine_path = engine_path
        self.default_scorer = partial(
            self.compute_single_score, time_limit=0.5
        )

    def setup(self, env=None):
        "Leggo"
        pass

    def __call__(self, states: TensorType["batch", 64]) -> Tensor:
        if isinstance(states, Board):
            return torch.tensor([self.default_scorer(states)])
        copied = states.copy()
        if torch.is_tensor(states):
            copied = copied.tolist()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            scores = executor.map(self.default_scorer, copied)
            executor.shutdown(wait=True, cancel_futures=False)
        return torch.tensor(list(scores))

    def compute_single_score(
        self,
        state: Board,
        time_limit: float,
    ) -> float:
        with SimpleEngine.popen_uci(self.engine_path) as eng:
            centipawn = eng.analyse(
                state, engine.Limit(time=time_limit), info=engine.INFO_SCORE
            )["score"].wdl().relative.expectation()
        return -centipawn
