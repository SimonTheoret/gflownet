"""Partial environment for chess"""

import re
from itertools import chain, product
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
        **kwargs,
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
        self.board = Board(fen) if fen is not None else Board()  # Board in a list

        self.eos = (-1, -1)  # End of sequence action

        self.fen_parser = FenParser()

        self.state = self.fen_parser.parse(
            self.board.fen(), self
        )  # parses the board's fen into a list containing the positions on the board

        self.source = self.state  # Source state

        super().__init__(**kwargs)

    def get_action_space(self) -> List:
        """
        Returns all possible actions. An action is defined as a destination and
        a move, i.e. as a pair of squares.

        Returns
        -------
        Returns the list containing all the possibles actions.
        """
        lis1 = [i for i in range(0, 64)]
        lis2 = [i for i in range(0, 64)]
        lis = list(product(lis1, lis2))
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
            return [True for _ in range(self.action_space_dim)]

        moves = [self._action_to_move(action) for action in possibles_actions]
        return [True if move not in self.board.legal_moves else False for move in moves]

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

        move = self._action_to_move(action)  # type: ignore

        # If action is eos or the game is over
        if action == self.eos or self.board.is_game_over():
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True  # type: ignore

        # If action is not eos and game is not over, perform action. This is
        # the main chunk !
        else:
            valid = True if move in self.board.legal_moves else False  # type: ignore
            if valid:
                # the state was internally updated in self._update_state
                self.n_actions += 1

                if self.board.gives_check(move):
                    self.board.push(move)
                    return self.state, self.eos, valid  # type: ignore

                self.board.push(move)

                self.state = self.fen_parser.parse(
                    self.board.fen(), self
                )  # update the state with the current fen

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
        return chess.Move(from_square=init_square, to_square=final_square)


class FenParser:
    def __init__(self):
        self.tokenizer = {
            "p": -1,
            "r": -2,
            "n": -3,
            "b": -4,
            "q": -5,
            "k": -6,
            "P": 1,
            "R": 2,
            "N": 3,
            "B": 4,
            "Q": 5,
            "K": 6,
            " ": 0,
        }

    def parse(self, fen_str: str, env: GFlowNetEnv) -> List[int]:
        """Parse a the fen_str into list (vector) of integers representing the
        board's positions.

        Parameters
        ----------
        fen_str: str
            The fen string parsed into a vector of integers. This is normally
            obtained by calling board.fen()

        Results
        -------
        Returns a list of integers, where each integer represents a piece or a
        empty square.
        """
        ranks = fen_str.split(" ")[0].split("/")
        pieces_on_all_ranks = [self.parse_rank(rank) for rank in ranks]
        flatten = list(
            chain(*pieces_on_all_ranks)
        )  # creates a flat version of the board
        return [self.tokenizer[i] for i in flatten] + [
            int(env.id)
        ]  # returns the tokenized version of the board

    def parse_rank(self, rank):
        rank_re = re.compile("(\d|[kqbnrpKQBNRP])")  # type: ignore
        piece_tokens = rank_re.findall(rank)
        pieces = self.flatten(map(self.expand_or_noop, piece_tokens))
        return pieces

    def flatten(self, lst):
        return list(chain(*lst))

    def expand_or_noop(self, piece_str):
        piece_re = re.compile("([kqbnrpKQBNRP])")
        retval = ""
        if piece_re.match(piece_str):
            retval = piece_str
        else:
            retval = self.expand(piece_str)
        return retval

    def expand(self, num_str):
        return int(num_str) * " "
