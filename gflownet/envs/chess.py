"""Environment for chess"""

from functools import partial
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
        # This attributes keeps track of what the next move is: Is it
        # a move (moving a piece) or is the move already decided and
        # we need to select the distance traveled by the piece.
        # self.move_or_distance = (
        #     Action.MOVE
        # )  # In the initial state, the next action should be a move.

        self.state = [
            0,
            0,
            self._fen_to_list(self.board),  # TODO: find representation for the board
        ]
        # The state is [int,int,list[int]], where the first int is
        # what player's turn it is, (white is 0, black is 1) the second int is the stage
        # (i.e. 0 for move and 1 for piece).
        self.eos = (-1, -1, -1)  # End of sequence action

    def get_action_space(self) -> List:
        """
        Constructs list with all possible actions, including eos. An
        action is represented by a tuple of length 3 (color, move or
        piece, position or piece). The colors can be 0 for whites and
        1 for black, the second position can either be 0 for piece or 1
        for move and the last position of the action tuple can be a
        position (with value in [0,63]) or piece (with value in
        [0,16]).

        """
        actions = []
        for color in (0, 1):
            for move_or_piece in (0, 1):
                if move_or_piece == 0:  # is a piece
                    for position_or_piece in range(0, 6):
                        actions.append((color, move_or_piece, position_or_piece))
                if move_or_piece == 1:  # is a move
                    for position_or_piece in range(0, 64):
                        actions.append((color, move_or_piece, position_or_piece))
        actions.append(self.eos)
        self.action_space_dim = len(actions)
        return actions

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

        possible_actions = self.get_action_space()  # list all possibles actions
        if done:
            return [
                True for _ in range(self.action_space_dim)
            ]  # If game is done, there is nothing to do!
        player, piece_or_move, fen = state
        board = self._parse_fen_list_to_board(fen)
        # mask moves for the other player and movement/piece selection:
        possible_actions = [
            action if action[0] == player and action[1] == piece_or_move else True
            for action in possible_actions
        ]

        # mask moves when not legal:
        possible_actions = [
            self._is_action_legal(board, action) # tells if move is legal it it is not a bool
            if not isinstance(action, bool)
            else action #if it is a bool, keep it there
            for action in possible_actions
        ]

        return possible_actions

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
        # _, self.state, action = self._pre_step(action, skip_mask_check)
        # return None, None, None
        pass

    # def _fen_to_list(self, board: Board):
    #     """
    #     Converts the board into a list of positions. Returns a list of integer.
    #     """
    #     epd_string = board.epd()
    #     list_int = []
    #     for i in epd_string:
    #         if i == " ":
    #             return list_int
    #         elif i != "/":
    #             if i in self.string2int:
    #                 list_int.append(self.string2int[i])
    #             else:
    #                 for _ in range(0, int(i)):
    #                     list_int.append(0)

    def _fen_to_list(self, board: Board) -> list[str]:
        """
        Returns the fen representation of a board as a list of characters. The
        length of this list will always be 100.

        Parameters
        ----------
        board: Board
            The board which representation is used to make the list of characters
        Returns
        -------
        The fen representation as a list of string with a padding of `A`s. This
        list will always be of length 100.
        """
        init = [*board.fen()]
        while len(init) < 100:
            init.append("A")  # Adds an arbitrary token at the end of the fen
            # string. This is to make sure the state is always of lenght 100.
        return init

    def _parse_fen_list_to_board(self, fen_list: List[str]) -> Board:
        """
        Parse a fen list into a board.

        Parameters
        ----------
        fen_list: List[str]
            list of characters, potentially containing padding. The padding is removed.

        Returns
        -------
        The board, in the state given by the fen representation.
        """
        fen = self._parse_fen_list(fen_list)
        return Board(fen)

    def _parse_fen_list(self, fen_list: List[str]) -> str:
        """
        Parses a fen list into a string. It removes the padding.

        Parameters
        ----------
        fen_list: List[str]
            list of characters, potentially containing padding. The padding is removed.

        Returns
        -------
        A string of the fen representation of the state, without the padding
        """
        idx = fen_list.index("A")  # finds the first padding element (`A`)
        fen_list = fen_list[:idx]  # take until the first `A` (excludes the first `A`)
        fen = "".join(
            fen_list
        )  # consumes the iterables and build it into a single string
        return fen

    def _convert_action_into_san(self, board: Board, action: tuple) -> str:
        """
        Converts an action into a san move.

        Parameters
        ----------
        board: Board
            Because san moves need context (i.e. the current state of the
                                            game), the board is needed to parse
            the move.

        """
        piece = chess.piece_symbol(chess.PieceType(action[1]))

        if action[0] == 0:  # uppercase if whites are playing
            piece = piece.upper() if piece is not None else None

        position = chess.SQUARE_NAMES[action[2]]  # Name of the square

        san = piece + position

        return san

    def _is_action_legal(self, board: Board, action: Tuple) -> bool:
        # TODO: description
        if isinstance(action, bool):
            return action
        san_action = self._convert_action_into_san(board, action)
        try:
            possible = board.parse_san(san_action) in board.legal_moves
            return possible
        except Exception:
            return False
