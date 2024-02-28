import pytest

from gflownet.envs.chess import GFlowChessEnv

ACTION_SPACE_SIZE = 140


@pytest.fixture
def env():
    return GFlowChessEnv()


@pytest.fixture
def env_default():
    return GFlowChessEnv()


def test_size_action_space():
    chess_env = env()
    assert len(chess_env.get_action_space()) == ACTION_SPACE_SIZE


def test_size_invalid_actions_mask():
    chess_env = env()
    assert len(chess_env.get_mask_invalid_actions_forward()) == ACTION_SPACE_SIZE
