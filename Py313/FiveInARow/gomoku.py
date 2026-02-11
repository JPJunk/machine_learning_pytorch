import hashlib
from typing import List, Optional, Tuple

import numpy as np

from utils import (
    GameResult,
    BOARD_SIZE,
    WIN_LENGTH,
    BLACK,
    WHITE,
    coord_to_index,
    index_to_coord,
)


# -----------------------------
# Gomoku environment
# -----------------------------

class Gomoku:
    """
    Core Gomoku environment.

    Responsibilities:
      - Track board state
      - Validate moves (empty intersection)
      - Track current player and move count
      - Check win condition (>= five in a row, all directions)
      - Provide legal moves
      - Reset & render
    """

    def __init__(self, size: int = BOARD_SIZE, win_len: int = WIN_LENGTH):
        self.size = size
        self.win_len = win_len
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = BLACK
        self.moves_played: int = 0
        self.last_move: Optional[Tuple[int, int]] = None
        self.result: GameResult = GameResult.ONGOING

    def reset(self) -> None:
        """Reset to initial state."""
        self.board.fill(0)
        self.current_player = BLACK  # Black always starts
        self.moves_played = 0
        self.last_move = None
        self.result = GameResult.ONGOING

    def hash(self) -> int:
        """
        Return a reproducible hash of the current board + player turn.

        Uses SHA-256 over the raw board bytes plus a single byte encoding
        the current player (0 for BLACK, 1 for WHITE).
        """
        player_byte = 0 if self.current_player == BLACK else 1
        # board is already int8 and contiguous; tobytes() is fine here
        data = self.board.tobytes() + bytes([player_byte])
        return int(hashlib.sha256(data).hexdigest(), 16)

    def get_valid_moves(self) -> List[int]:
        """Return list of legal move indices (all empty intersections)."""
        rows, cols = np.where(self.board == 0)
        return [coord_to_index(r, c, self.size) for r, c in zip(rows, cols)]

    def make_move(self, action: int) -> bool:
        """
        Place a stone for the current player at `action` (flat index).

        Returns:
            True if the move was legal and applied, False otherwise.
        """
        r, c = divmod(action, self.size)
        if self.board[r, c] != 0:
            return False

        # 1) place stone
        self.board[r, c] = self.current_player
        # 2) update last_move and moves_played
        self.last_move = (r, c)
        self.moves_played += 1
        # 3) check result NOW, before toggling
        self.result = self.check_result()
        # 4) toggle player only if game continues
        if self.result == GameResult.ONGOING:
            self.current_player = WHITE if self.current_player == BLACK else BLACK
        return True

    def undo_move(self) -> bool:
        """
        Undo the last move and restore the previous player.

        Note: this only undoes a single move and clears last_move.
        """
        if self.last_move is None:
            return False
        r, c = self.last_move
        self.board[r, c] = 0
        self.moves_played -= 1
        # Flip back to the player who made the undone move
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        self.last_move = None
        self.result = GameResult.ONGOING
        return True

    def clone(self) -> "Gomoku":
        """Return a deep copy of the current game state."""
        new_env = Gomoku(self.size, self.win_len)
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.moves_played = self.moves_played
        new_env.last_move = self.last_move
        new_env.result = self.result
        return new_env

    def check_result(self) -> GameResult:
        """
        Check the game result based on the last move.

        Returns:
            GameResult.BLACK_WIN / WHITE_WIN / DRAW / ONGOING
        """
        if self.last_move is None:
            return GameResult.ONGOING

        r, c = self.last_move
        player = self.board[r, c]
        if player == 0:
            return GameResult.ONGOING

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        def count_dir(dr: int, dc: int) -> int:
            cnt = 1
            rr, cc = r + dr, c + dc
            while (
                0 <= rr < self.size
                and 0 <= cc < self.size
                and self.board[rr, cc] == player
            ):
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while (
                0 <= rr < self.size
                and 0 <= cc < self.size
                and self.board[rr, cc] == player
            ):
                cnt += 1
                rr -= dr
                cc -= dc
            return cnt

        for dr, dc in directions:
            length = count_dir(dr, dc)
            if length >= self.win_len:
                return GameResult.BLACK_WIN if player == BLACK else GameResult.WHITE_WIN

        if self.moves_played >= self.size * self.size:
            return GameResult.DRAW

        return GameResult.ONGOING

    def find_any_five(self, win_len: int = WIN_LENGTH):
        """
        Find any sequence of `win_len` stones for either player.

        Returns:
            (player, coords) where player is BLACK/WHITE and coords is a list
            of (row, col) positions, or (None, []) if none found.
        """
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                player = self.board[r, c]
                if player == 0:
                    continue
                for dr, dc in dirs:
                    coords = [(r, c)]
                    rr, cc = r + dr, c + dc
                    while (
                        0 <= rr < self.size
                        and 0 <= cc < self.size
                        and self.board[rr, cc] == player
                    ):
                        coords.append((rr, cc))
                        if len(coords) >= win_len:
                            return player, coords
                        rr += dr
                        cc += dc
        return None, []

    def render(self) -> None:
        """Simple text rendering (X for Black, O for White, . for empty)."""
        rows = []
        for r in range(self.size):
            row_chars = []
            for c in range(self.size):
                val = self.board[r, c]
                if val == BLACK:
                    row_chars.append("X")
                elif val == WHITE:
                    row_chars.append("O")
                else:
                    row_chars.append(".")
            rows.append(" ".join(row_chars))
        print("\n".join(rows))
        print(
            f"Current player: {'BLACK(X)' if self.current_player == BLACK else 'WHITE(O)'}"
        )