"""Environment for chess"""

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
        # This attributes keeps track of what the next move is: Is it a move
        # (moving a piece) or is the move already decided and we need to select
        # the distance traveled by the piece. self.move_or_distance = (
        # Action.MOVE)  # In the initial state, the next action should be a
        # move.

        self.state = [0, 0, self._fen_to_list(self.board), -1]
        # The state is [int, int, list[int], int], where the first int is
        # what player's turn it is, (white is 0, black is 1) the second int is the stage
        # (i.e. 0 for selecting the piece and 1 for moving it), the third element is fen
        # representation of the board (in a list) and the last int is what is
        # the current piece selected (can be from 1 to 6). -1 state[3] indicates that no piece have been selected.
        self.eos = (-1, -1, -1)  # End of sequence action

    def get_action_space(self) -> List:
        """
        Constructs list with all possible actions, including eos. An
        action is represented by a tuple of length 3 (color, move or
        piece, position or piece). The colors can be 0 for whites and
        1 for black, the second position can either be 0 for piece or 1
        for move and the last position of the action tuple can be a
        position (with value in [0,63]) or piece (with value in
        [1,6]).

        """
        actions = []
        for color in (0, 1):
            for move_or_piece in (0, 1):
                if move_or_piece == 0:  # is a piece
                    for position_or_piece in range(1, 7):
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

        player, piece_or_move, fen, current_piece = state
        board = self._parse_fen_list_to_board(fen)

        # mask moves for the other player and movement/piece selection:
        possible_actions = [
            action if action[0] == player and action[1] == piece_or_move else True
            for action in possible_actions
        ]

        # Treat move and piece selection differently:

        # Case for selecting a piece
        if piece_or_move == 0:  # when selecting the piece to move
            possible_actions = [
                self._piece_is_dead(state, action)
                if not isinstance(action, bool)
                else action  # if it is a bool, keep it there
                for action in possible_actions
            ]
        # Case for making the move
        if piece_or_move == 1:
            possible_actions = [
                self._action_is_illegal(
                    state, board, action
                )  # tells if move is legal it it is not a bool
                if not isinstance(action, bool)
                else action  # if it is a bool, keep it there
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
            return self.state, self.eos, True # type: ignore
        # If action is not eos, then perform action
        else:
            move = self._convert_action_into_san(state = self.state, action = action)
            try:
                self.board.push_san(move) #updates the board with the move
                valid = True
            except Exception:
                valid = False

            state_next = self._fen_to_list(self.board)
            if valid:
                self.state = state_next
                self.n_actions += 1
            return self.state, action, valid



    def _fen_to_list(self, board: Board) -> list[str]:
        """
        Returns the fen representation of a board as a list of characters. The
        length of this list will always be 100.

        Parameters
        ----------
        board: Board
            The board which representation is used to make the list of
            characters.

        Returns
        -------
        The fen representation as a list of string with a padding of `A`s. This
        list will always be of length 100. This representation of the board is
        used in the state.
        """
        init = [*board.fen()]
        while len(init) < 100:
            init.append("A")  # Adds an arbitrary token at the end of the fen
            # string. This is to make sure the state is always of lenght 100
            # and is therefore a valid input for our model
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

    def _piece_to_str(self, color: int, piece: int, include_pawns: bool = False) -> str:
        """
        Converts a piece to a symbol, given its color. Returns the letter
        representing the piece. For example, the pawn are represented with the
        empty string if include_pawns is false, while white's king is `K`,
        black's king `k`, white's knight is `N`, etc. If `include_pawns =
        True`, and the piece is a pawn (`piece = 1`), returns `p` or `P`.

        Parameters
        ----------
        color: int
            Color of the piece. Can be 0 for white or 1 for black. Used to
            specify the case of the returned string.
        piece: int
            Can be any integer in {1,2,3,4,5,6}.
        include_pawns: bool, Default False
            When `True`, this means the returned string can contain `p` or `P`
            for pawn, if the piece is a pawn. Should be `False` when used with
            `push_san()`, as this method does not expect the pawn to be
            specified.

        Returns
        -------
        Returns the string representation of the piece or possibly the empty string for the pawn.
        """
        piece_str = ""

        if (
            piece != 1 or include_pawns
        ):  # will only be skipped if the piece is a pawn and we must not include pawns
            piece_str = chess.piece_symbol(chess.PieceType(piece))

            if color == 0:  # uppercase if whites are playing
                piece_str = piece_str.upper()

            if color == 1:  # uppercase if blacks are playing
                piece_str = piece_str.lower()

        return piece_str

    def _convert_action_into_san(self, state: List, action: tuple) -> str:
        """
        Converts an action into a san move.

        Parameters
        ----------
        state: list
            The current state of the game.
        action: Tuple
            The action converted to san (standard algebraic notation).

        Returns
        -------
        Returns a string, representing the san notation of the move. This is
        achieved by parsing the san with the help of `board.parse_san()`.
        """

        color = action[0]
        destination = action[2]
        piece = state[3]  # the piece for the current state is preserved in the state
        piece = self._piece_to_str(color=color, piece=piece)

        position = chess.SQUARE_NAMES[destination]  # Name of the square

        san = piece + position
        return san

    def _action_is_illegal(self, state: List, board: Board, action: Tuple) -> bool:
        """
        Tests if action is illegal given a board. If the action is illegal
        (i.e. not in `board.legal_moves`), returns True.

        Parameters
        ----------
        state: List
            The current state of the game. Used for converting the action into a san.
        board: Board
            The board for which the legality of the action is tested.
        action: Tuple
            The action tested for legality.

        Returns
        -------
        Returns `True` if the action is illegal or ambiguous given the board, and `False` otherwise.
        """
        if isinstance(action, bool):  # if 'action' is a bool, simply return it
            return action
        san_action = self._convert_action_into_san(
            state, action
        )  # convert action into san
        try:  # we use try because some move are legal but ambiguous
            return board.parse_san(san_action) not in board.legal_moves
        except (
            Exception
        ):  # parse_san raise an exception, simply declare the move illegal
            return True

    def _piece_is_dead(self, state: List, action: Tuple) -> bool:
        """
        Tests whether or not the piece selected by the action is alive (i.e. still on the board).

        Parameters
        ----------
        state: List
            State of the current game.
        action: Tuple
            Action specifying the piece getting moved.

        Returns
        -------
        Returns True if the piece contained in the action is dead (i.e. not on the board) and False otherwise.
        """
        fen = state[
            2
        ]  # fen is the result of _fen_to_list, and therefore should be of size 100
        piece = state[3]
        assert piece in range(1, 7)
        piece_str = self._piece_to_str(color=action[0], piece=piece)
        return piece_str not in fen

    def _update_state(self, action: Tuple) -> Tuple[List, bool]:
        """
        Update the states with the action.

        Parameters
        ----------
        action: Tuple
            Action to (try to) apply. Modifies the current state.

        Returns
        -------
        Returns a tuple containing the updated state at index 0 and a boolean
        specifying if the action is valid or not.
        """
        if action[1] == 0:
            pass
        if action[1] == 1:
            pass
        pass
