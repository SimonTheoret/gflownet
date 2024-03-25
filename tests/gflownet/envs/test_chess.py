import pytest
import chess
from chess import Board

from gflownet.envs.chess_class import GFlowChessEnv

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

def test_get_parents(env: GFlowChessEnv):
    env = GFlowChessEnv()
    env.state = Board()
    action = (8,16) # move a pawn forward
    move = env._action_to_move(action)
    env.state.push(move)
    actual = env.get_parents()
    print(actual)
    print(env.state)
    assert actual == ([Board()], [(8,16)])
