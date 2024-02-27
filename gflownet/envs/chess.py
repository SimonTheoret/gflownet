"""Environment for chess"""

from gflownet.envs.base import GFlowNetEnv
from typing import Tuple, List, Optional
from chess import Board


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

        self.mapped = {
            # Map of the EPD representation to number. Minus for
            # blacks, positive for whites and 0 for empty positions.
            "P": 1,  # White Pawn
            "p": -1,  # Black Pawn
            "N": 2,  # White Knight
            "n": -2,  # Black Knight
            "B": 3,  # White Bishop
            "b": -3,  # Black Bishop
            "R": 4,  # White Rook
            "r": -4,  # Black Rook
            "Q": 5,  # White Queen
            "q": -5,  # Black Queen
            "K": 6,  # White King
            "k": -6,  # Black King
            ".": 0,  # Empty position
        }
        self.state = [
            0,
            0,
            self._convert_to_list(),  # TODO: find representation for the board
        ]
        # The state is [int,int,list[int]], where the first int is
        # what player's turn it is, the second int is the stage
        # (i.e. 0 for move and 1 for piece).

    def get_action_space(self):
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
                    for position_or_piece in range(0, 16):
                        actions.append((color, move_or_piece, position_or_piece))
                if move_or_piece == 1:  # is a move
                    for position_or_piece in range(0, 64):
                        actions.append((color, move_or_piece, position_or_piece))

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
        # TODO: Convert action to san moves
        pass

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
        _, self.state, action = self._pre_step(action, skip_mask_check)
        return None, None, None

    def _convert_to_list(self):
        """
        Converts the board EPD into a matrix of positions. Returns a list of integer.
        """
        epd_string = self.board.epd()
        list_int = []
        for i in epd_string:
            if i == " ":
                return list_int
            elif i != "/":
                if i in self.mapped:
                    list_int.append(self.mapped[i])
                else:
                    for _ in range(0, int(i)):
                        list_int.append(0)
