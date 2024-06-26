"""Partial environment for chess"""

import re
from itertools import chain, product
from typing import List, Optional, Tuple

from torchtyping import TensorType
import chess
import torch
from chess import Board
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv

chess.Board.__len__ = lambda x: 1


class GFlowChessEnv(GFlowNetEnv):
    """
    Environment for the GFlowChess.
    """

    def __init__(
        self,
        fen: Optional[str] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
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
        self.state: Board = Board(fen) if fen is not None else Board()  # Board

        self.eos = (-1, -1)  # End of sequence action

        self.fen_parser = FenParser()

        self.source: Board = Board(fen) if fen is not None else Board()

        # initial count of actions
        self.n_actions = 0

        # How many actions can we do in a single sequence (maximum length of the sequence)
        self.max_n_actions = 20

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
        state=None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        """
        if state is None:
            state: Board = self.state

        if done is None:
            done: bool = self.done

        # Done case. Nothing to do
        if done:
            return [True for _ in range(self.action_space_dim)]

        possibles_actions = self.get_action_space()

        # if sequence is completed or the game is over
        if isinstance(state, Board):
            if state.is_game_over() or self.n_actions >= self.max_n_actions:
                return [
                    True if action != self.eos else False
                    for action in possibles_actions
                ]

            # if sequence is not completed and the done is false:
            moves = [
                self._action_to_move(action) if action != self.eos else action
                for action in possibles_actions
            ]
            return [
                True if move == self.eos or move not in state.legal_moves else False
                for move in moves
            ]
        else:
            print("The state is not a board!!!")
            print("state is type: ")
            print(type(state))

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

        # If action is eos or the game is over
        if action == self.eos:
            self.done = True
            self.n_actions += 1

            return self.state, self.eos, True  # type: ignore

        move = self._action_to_move(action)
        # If action is not eos, perform action. This is the main
        # chunk!
        if isinstance(self.state, Board):
            valid = move in self.state.legal_moves
            if valid:
                # the state was internally updated in self._update_state
                self.state = self.state.copy()
                self.n_actions += 1
                self.state.push(move)

                return self.state, action, valid
            else:
                return self.state, action, valid
        else:
            print("State is not a board!")
            print("State type:")
            print(type(self.state))

    def _action_to_move(self, action: Tuple[int]) -> chess.Move:
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
        final_square = chess.SQUARES[action[1]]  # type: ignore
        return chess.Move(from_square=init_square, to_square=final_square)

    # DONE: implement state2policy

    def _state2policy(self, state) -> torch.Tensor:
        return self.fen_parser.tokenize_chess_board(board=state, env=self)

    def states2policy(self, states) -> torch.Tensor:
        parsed_board_list = []
        for state in states:
            parsed_board_list.append(self._state2policy(state))
        stacked = torch.stack(parsed_board_list)
        return stacked

    # DONE: implement state2proxy
    def _state2proxy(
        self, state: List | TensorType["state_dim"] = None
    ) -> TensorType["state_proxy_dim"]:
        if torch.is_tensor(state):
            print("what the hell state is a tensor !!")
        else:
            return state

    # DONE: implement states2proxy
    def states2proxy(
        self, states: List | TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        if torch.is_tensor(states):
            print("what the hell state is a tensor!!")
        else:
            return states

    def get_parentss(self, state=None, done=None, action=None):
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []
        copy_state = state.copy()
        action = copy_state.pop()
        action = (int(action.from_square), int(action.to_square))
        return [copy_state], [action]

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> tuple[List, List]:
        current_board = state.copy() if state is not None else self.state.copy()
        exact_parent_state = current_board.copy()
        exact_parent_move = exact_parent_state.pop()
        if current_board.turn == chess.WHITE:
            current_board.turn = chess.BLACK
        else:
            current_board.turn = chess.WHITE
        if state is None:
            state = self.state
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        # if action == self.eos: #NOTE: needed?
        #     return [state], [self.eos]
        if state == self.source:
            return [], []
        # remove all move that could lead to a capture and pawn movements
        non_pawn_moves = self.legal_moves_without_capture_and_pawn_moves(current_board)
        missing_pieces_opponents = self.get_missing_pieces_by_type(current_board)
        pawn_moves = self.generate_pawn_moves(current_board, missing_pieces_opponents)
        resulting_boards = []
        resulting_moves = []

        # FIXME: This should not be necessary
        resulting_boards.append(exact_parent_state)
        resulting_moves.append(
            (exact_parent_move.from_square, exact_parent_move.to_square)
        )

        # generate board from movements
        for move in non_pawn_moves:
            board_copy = current_board.copy()
            board_copy.push(move)
            resulting_boards.append(board_copy.copy())
            resulting_moves.append((move.to_square, move.from_square))
            # take into consideration all pieces that could have been eaten
            for piece in missing_pieces_opponents.keys():
                board_copy.set_piece_at(
                    move.from_square, chess.Piece.from_symbol(piece)
                )
                resulting_boards.append(board_copy.copy())
                resulting_moves.append((move.to_square, move.from_square))
                board_copy.remove_piece_at(move.from_square)
        # generate pawn movements
        for move in pawn_moves:
            board_copy = current_board.copy()
            board_copy.push(move)
            # check if pawn is moving diagonal (eating)
            if move.from_square % 8 != move.to_square % 8:
                for piece in missing_pieces_opponents.keys():
                    board_copy.set_piece_at(
                        move.from_square, chess.Piece.from_symbol(piece)
                    )
                    resulting_boards.append(board_copy.copy())
                    resulting_moves.append((move.to_square, move.from_square))
                    board_copy.remove_piece_at(move.from_square)
            else:
                resulting_boards.append(board_copy.copy())
                resulting_moves.append((move.to_square, move.from_square))
        return resulting_boards, resulting_moves

    def legal_moves_without_capture_and_pawn_moves(self, board):
        non_capture_moves = []

        for move in board.pseudo_legal_moves:
            # Check if the move results in a capture
            if (
                not board.is_capture(move)
                and board.piece_at(move.from_square).piece_type != chess.PAWN
            ):
                non_capture_moves.append(move)

        return non_capture_moves

    def get_missing_pieces_by_type(self, board):
        opponent_color = not board.turn
        # piece normally on a chess board at the beginning
        starting_piece_count = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1}
        # white piece are in capital
        if opponent_color:
            starting_piece_count = {
                key.upper(): value for key, value in starting_piece_count.items()
            }
        # Create a set to store all opponent pieces currently on the board
        opponent_pieces_on_board = {}

        # Iterate over all squares on the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == opponent_color:
                if piece.symbol() not in opponent_pieces_on_board:
                    opponent_pieces_on_board[piece.symbol()] = 0
                opponent_pieces_on_board[piece.symbol()] += 1

        # Update the missing pieces count
        for piece_type, count in opponent_pieces_on_board.items():
            starting_piece_count[piece_type] -= opponent_pieces_on_board[piece_type]
            # If all pieces are currently on the board, delete the key
            if starting_piece_count[piece_type] <= 0:
                starting_piece_count.pop(piece_type)

        return starting_piece_count

    def generate_pawn_moves(self, board, missing_pieces_opponents):
        """Take all the missing pieces and make all the possibles pawn moves."""
        previous_pawn_moves = []

        # Iterate over all squares on the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if (
                piece is not None
                and piece.piece_type == chess.PAWN
                and piece.color == board.turn
            ):
                # Get possible previous squares for the pawn
                previous_squares = []
                if piece.color == chess.WHITE:
                    if (
                        square >= 16 and board.piece_at(square - 8) is None
                    ):  # If pawn has moved at least one square forward
                        previous_squares.append(square - 8)  # Move one square back
                    if (
                        31 >= square >= 24
                        and board.piece_at(square - 8) is None
                        and board.piece_at(square - 16) is None
                    ):
                        # If pawn is two squares ahead of initial position
                        previous_squares.append(square - 16)  # Move two squares back
                elif piece.color == chess.BLACK:
                    if (
                        square < 48 and board.piece_at(square + 8) is None
                    ):  # If pawn has moved at least one square forward
                        previous_squares.append(square + 8)  # Move one square back
                    if (
                        39 >= square >= 32
                        and board.piece_at(square + 8) is None
                        and board.piece_at(square + 16) is None
                    ):
                        # If pawn is two squares ahead of initial position
                        previous_squares.append(square + 16)  # Move two squares back

                # Generate moves for each possible previous square
                for prev_sq in previous_squares:
                    previous_pawn_moves.append(chess.Move(square, prev_sq))

                # Check for diagonal captures if opponent pieces are missing
                if missing_pieces_opponents:
                    # checks if clear behind and if not in second row
                    if piece.color == chess.WHITE:
                        if (
                            square + 1 % 8 != 0
                            and square - 7 >= 8
                            and board.piece_at(square - 7) is None
                        ):
                            previous_pawn_moves.append(chess.Move(square, square - 7))
                        if (
                            square % 8 != 0
                            and square - 9 >= 8
                            and board.piece_at(square - 9) is None
                        ):
                            previous_pawn_moves.append(chess.Move(square, square - 9))
                    elif piece.color == chess.BLACK:
                        if (
                            square % 8 != 0
                            and square + 7 <= 55
                            and board.piece_at(square + 7) is None
                        ):
                            previous_pawn_moves.append(chess.Move(square, square + 7))
                        if (
                            square + 1 % 8 != 0
                            and square + 9 <= 55
                            and board.piece_at(square + 9) is None
                        ):
                            previous_pawn_moves.append(chess.Move(square, square + 9))

        return previous_pawn_moves


class FenParser:
    def __init__(self):
        one_hot_classes = torch.nn.functional.one_hot(torch.arange(0, 13)).type(
            torch.FloatTensor
        )
        self.tokenizer = {
            "p": one_hot_classes[0],
            "r": one_hot_classes[1],
            "n": one_hot_classes[2],
            "b": one_hot_classes[3],
            "q": one_hot_classes[4],
            "k": one_hot_classes[5],
            "P": one_hot_classes[6],
            "R": one_hot_classes[7],
            "N": one_hot_classes[8],
            "B": one_hot_classes[9],
            "Q": one_hot_classes[10],
            "K": one_hot_classes[11],
            " ": one_hot_classes[12],
        }  # DONE: Change this encoding for a one-hot encoding

    def parse(self, fen_str: str) -> List[torch.Tensor]:
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
        return [self.tokenizer[i] for i in flatten]
        # returns the tokenized version of the board

    def parse_rank(self, rank):
        rank_re = re.compile(r"(\d|[kqbnrpKQBNRP])")  # type: ignore
        piece_tokens = rank_re.findall(rank)
        pieces = self.flatten(map(self.expand_or_noop, piece_tokens))
        return pieces

    def pretty_print_board(self, fen_str: str):
        ranks = fen_str.split(" ")[0].split("/")
        pieces_on_all_ranks = [self.parse_rank(rank) for rank in ranks]
        return pieces_on_all_ranks

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

    def tokenize_chess_board(self, board: Board, env: GFlowChessEnv) -> torch.Tensor:
        """
        Tokenize a chess board using the FEN representation and return as a PyTorch tensor.

        Parameters:
        - board (Board): A chess board object.
        - env (GFlowNetEnv): The environment.

        Returns:
        - torch.Tensor: A PyTorch tensor representing the tokenized board.
        """
        fen_str = board.fen()
        parser = env.fen_parser
        tokenized_board = parser.parse(fen_str)
        return torch.cat(tokenized_board)
