import pytest
import chess
from chess import Board

from gflownet.envs.chess_class import FenParser, GFlowChessEnv
from tests.gflownet.envs import common

ACTION_SPACE_SIZE = 4097


@pytest.fixture
def env():
    return GFlowChessEnv()


@pytest.fixture
def env_default():
    return GFlowChessEnv()

#@pytest.mark.parametrize("env",['env'])
# def test__current(env):
#     chess_env = env
#     test_size_action_space(chess_env)
#     test_size_invalid_actions_mask(chess_env)
#     test_parser(chess_env)
#     test_step_board(chess_env)


def test_size_action_space(env: GFlowChessEnv):
    chess_env = env
    assert len(chess_env.get_action_space()) == ACTION_SPACE_SIZE


def test_size_invalid_actions_mask(env: GFlowChessEnv):
    chess_env = env
    assert len(chess_env.get_mask_invalid_actions_forward()) == ACTION_SPACE_SIZE

def parse(board: Board, env: GFlowChessEnv):
    parser = FenParser()
    return parser.tokenize_chess_board(board, env)

def test_get_parents(env: GFlowChessEnv):
    env = GFlowChessEnv()
    env.state = Board(fen = "rnbqkbnr/ppp1pppp/3p4/3P4/8/8/PPP1PPPP/RNBQKBNR")
    print(env.state.turn)
    # action = (1,16) # move a pawn forward
    # move = env._action_to_move(action)
    # env.state.push(move)
    actual = env.get_parents()
    for board in actual[0]:
        print(board)
        print("\n")
    print(actual[1])
    print(f"Length of the parents list: {len(actual[0])}")
    assert actual == ([Board()], [(8,16)])

class TestChessCommon(common.BaseTestsDiscrete):
    """Common tests for Chess."""
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob":100,
        }
        self.n_states = {}  # TODO: Populate.
