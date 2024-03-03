import pytest

from gflownet.envs.chess import GFlowChessEnv

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


def test_size_action_space(env):
    chess_env = env
    assert len(chess_env.get_action_space()) == ACTION_SPACE_SIZE


def test_size_invalid_actions_mask(env):
    chess_env = env
    assert len(chess_env.get_mask_invalid_actions_forward()) == ACTION_SPACE_SIZE


def test_parser(env):
    chess_env = env
    assert chess_env.fen_parser(chess_env.board) == [-2, -3, -4, -5, -6, -4, -3, -2, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 4, 3, 2]

def test_step_board(env):
    chess_env = env
    # valid move
    assert chess_env.step((1,16)) == ([-2, -3, -4, -5, -6, -4, -3, -2, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 4, 5, 6, 4, 3, 2], (1, 16), True)
    assert chess_env.step((48,32)) == ([-2, -3, -4, -5, -6, -4, -3, -2, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 4, 5, 6, 4, 3, 2], (48, 32), True)

    # invalid move
    assert chess_env.step((0,0)) == ([-2, -3, -4, -5, -6, -4, -3, -2, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 4, 5, 6, 4, 3, 2], (48, 32), False)
    



def test_convertion_into_san():
    pass

