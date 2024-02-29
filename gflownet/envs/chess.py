"""Partial environment for chess"""

from itertools import combinations
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
        self.state = [Board(fen) if fen is not None else Board()]  # Board in a list
        self.eos = (-1, -1)  # End of sequence action

    def get_action_space(self) -> List:
        """
        Returns all possible actions. An action is defined as a destination and
        a move, i.e. as a pair of squares.

        Returns
        -------
        Returns the list containing all the possibles actions.
        """
        lis = list(combinations([i for i in range(0,64)],2))
        lis.append(self.eos)
        return lis


    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        """
        if state is None:
            state = self.state

        if done is None:
            done = self.done
        possibles_actions = self.get_action_space()
        if done:
            return  [True for _ in range(self.action_space_dim)]

        moves =  [self._action_to_move(action) for action in possibles_actions]
        return [True if move not in state[0].legal_moves else False for move in moves]

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
            The state after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """

        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        # If action is eos
        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True  # type: ignore

        # If action is not eos, then perform action. This is the main chunk !
        else:
            move = self._action_to_move(action) #type: ignore
            valid = True if move in self.state[0].legal_moves else False # type: ignore
            if valid:
                # the state was internally updated in self._update_state
                self.n_actions += 1
                self.state[0].push(move) # type: ignore
            return self.state, action, valid


    def _action_to_move(self, action: Tuple[int, int]) -> chess.Move:
        """
        Transform an action into a chess.Move object. It treats the first
        integer as the initial square and the second integer as the destination

        Parameters
        ----------
        Action: Tuple[int, int],
            Action for making a move.

        Returns
        -------
        Returns a chess.Move object representing the action.
        """
        init_square = chess.SQUARES[action[0]]
        final_square = chess.SQUARES[action[1]]
        return chess.Move(from_square= init_square, to_square = final_square)
