"""Partial environment for chess"""

from typing import List, Optional, Tuple

import chess
from chess import Board

from gflownet.envs.base import GFlowNetEnv


class GFlowChessEnv(GFlowNetEnv):
    """
    Environment for the GFlowChess.
    """

    def __init__(
        self,
        fen: Optional[str] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ):
        """
        Initialize the State space.

        Parameters
        ----------
        initial_position: str.
            Initial position on the board. By default, it uses a
            traditional, full board with fen value:
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

        """
        self.board = Board(fen) if fen is not None else Board()  # Board
        self.eos = (-1, -1)  # End of sequence action

    def get_action_space(self) -> List:
        pass

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        pass
        # """
        # Returns a list of length the action space with values:
        #     - True if the forward action is invalid from the current state.
        #     - False otherwise.
        # """
        #
        # if state is None:
        #     state = self.state
        #
        # if done is None:
        #     done = self.done
        #
        # possible_actions = self.get_action_space()  # list all possibles actions
        # if done:
        #     return [
        #         True for _ in range(self.action_space_dim)
        #     ]  # If game is done, there is nothing to do!
        #
        # player, piece_or_move, fen, current_piece = state
        # board = self._parse_fen_list_to_board(fen)
        #
        # # mask moves for the other player and movement/piece selection:
        # possible_actions = [
        #     action if action[0] == player and action[1] == piece_or_move else True
        #     for action in possible_actions
        # ]
        #
        # # Treat move and piece selection differently:
        #
        # # Case for selecting a piece
        # if piece_or_move == 0:  # when selecting the piece to move
        #     possible_actions = [
        #         self._piece_is_dead(state, action)
        #         if not isinstance(action, bool)
        #         else action  # if it is a bool, keep it there
        #         for action in possible_actions
        #     ]
        # # Case for making the move
        # if piece_or_move == 1:
        #     possible_actions = [
        #         self._action_is_illegal(
        #             state, board, action
        #         )  # tells if move is legal it it is not a bool
        #         if not isinstance(action, bool)
        #         else action  # if it is a bool, keep it there
        #         for action in possible_actions
        #     ]
        # return possible_actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        pass
        # # Generic pre-step checks
        # do_step, self.state, action = self._pre_step(
        #     action, skip_mask_check or self.skip_mask_check
        # )
        # if not do_step:
        #     return self.state, action, False
        # # If action is eos
        # if action == self.eos:
        #     self.done = True
        #     self.n_actions += 1
        #     return self.state, self.eos, True  # type: ignore
        #
        # # If action is not eos, then perform action. This is the main chunk !
        # else:
        #     move = self._convert_action_into_san(state=self.state, action=action)
        #     try:
        #         self.board.push_san(move)  # updates the board with the move
        #         valid = True
        #         self._update_state(action) # type: ignore
        #     except Exception:
        #         valid = False
        #
        #     if valid:
        #         # the state was internally updated in self._update_state
        #         self.n_actions += 1
        #     return self.state, action, valid

