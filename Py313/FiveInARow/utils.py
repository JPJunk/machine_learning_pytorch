import enum
from typing import Tuple

import numpy as np
import torch

# -----------------------------
# Constants and configs
# -----------------------------

BOARD_SIZE = 15
WIN_LENGTH = 5

# Encodings:
#  0 = empty, 1 = black (X), -1 = white (O)
BLACK = 1
WHITE = -1
# EMPTY = 0  # you can uncomment if you want a named constant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Core data structures
# -----------------------------

class Transition:
    """
    Experience tuple for replay buffer and AlphaZero-style training.

    Attributes:
        state: board array before action (e.g. np.ndarray of shape [H, W])
        action: integer index of move taken (0..board_size^2-1)
        reward: scalar reward (post-game shaping)
        next_state: board array after action
        done: bool, whether this transition ends the game
        player: BLACK or WHITE who took the action
        q_value: optional Q estimate (from MCTS or value net)
        pi: optional MCTS visit distribution (np.ndarray length board_size^2)
        z: optional final outcome from this player's perspective (+1/-1/0)
        meta: optional dict for extra info (e.g. mode, flags)
    """
    def __init__(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        player,
        q_value: float = 0.0,
        pi=None,
        z=None,
        meta=None,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.player = player
        self.q_value = q_value
        self.pi = pi
        self.z = z
        self.meta = meta or {}


class GameResult(enum.Enum):
    """Outcome of a finished game."""
    BLACK_WIN = enum.auto()
    WHITE_WIN = enum.auto()
    DRAW = enum.auto()
    ONGOING = enum.auto()


# -----------------------------
# Coordinate helpers
# -----------------------------

def index_to_coord(action: int, size: int = BOARD_SIZE) -> Tuple[int, int]:
    """Map flat action index to (row, col)."""
    return divmod(action, size)


def coord_to_index(row: int, col: int, size: int = BOARD_SIZE) -> int:
    """Map (row, col) to flat action index."""
    return row * size + col


# -----------------------------
# Pattern / threat detection
# -----------------------------

def detect_pattern(board: np.ndarray, player: int, length: int = 4, open_ends: int = 2) -> bool:
    """
    Detect sequences of `length` stones for `player` with `open_ends` empty spaces.

    Example: open-four = XXXX with both ends empty:
        [0, player, player, player, player, 0]

    Returns:
        True if at least one such pattern exists anywhere on the board.
    """
    size = board.shape[0]

    # Helper: check a 1D line
    def check_line(line: np.ndarray) -> bool:
        # window length = length + open_ends (e.g. 4 + 2 = 6 for open-four)
        win_len = length + open_ends
        if len(line) < win_len:
            return False
        for i in range(len(line) - win_len + 1):
            window = line[i:i + win_len]
            # Example open-four: [0, p, p, p, p, 0]
            if (
                window[0] == 0
                and window[-1] == 0
                and np.all(window[1:-1] == player)
            ):
                return True
        return False

    # Rows
    for r in range(size):
        if check_line(board[r, :]):
            return True
    # Cols
    for c in range(size):
        if check_line(board[:, c]):
            return True
    # Diagonals
    for offset in range(-size + 1, size):
        if check_line(np.diagonal(board, offset=offset)):
            return True
        if check_line(np.diagonal(np.fliplr(board), offset=offset)):
            return True

    return False


def detect_double_open_three(board: np.ndarray, player: int) -> bool:
    """
    Detect if the board contains two or more distinct open-three patterns for `player`.

    Open-three: XXX with both ends open (e.g. .XXX.).

    Returns:
        True if >= 2 such patterns exist anywhere on the board.
    """
    size = board.shape[0]
    count = 0

    def check_line(line: np.ndarray) -> None:
        nonlocal count
        # window of 5: [0, p, p, p, 0]
        if len(line) < 5:
            return
        for i in range(len(line) - 5 + 1):
            window = line[i:i + 5]
            if (
                window[0] == 0
                and window[-1] == 0
                and np.all(window[1:-1] == player)
            ):
                count += 1

    # Rows
    for r in range(size):
        check_line(board[r, :])
    # Cols
    for c in range(size):
        check_line(board[:, c])
    # Diagonals
    for offset in range(-size + 1, size):
        check_line(np.diagonal(board, offset=offset))
        check_line(np.diagonal(np.fliplr(board), offset=offset))

    return count >= 2


def creates_open_four(board: np.ndarray, player: int, win_len: int = WIN_LENGTH) -> bool:
    """
    Return True if board contains an open-four (XXXX with both ends open).

    This is a shape-based threat detector; it does not require the game
    to be in a winning state yet.
    """
    # For standard Gomoku, open-four is length=4 with 2 open ends.
    return detect_pattern(board, player, length=4, open_ends=2)

def creates_double_three(board: np.ndarray, player: int, win_len: int = WIN_LENGTH) -> bool:
    """
    Return True if board contains two simultaneous open-threes for `player`.

    This is a classic double-three threat: two .XXX. patterns.
    """
    return detect_double_open_three(board, player)

# def detect_four_in_a_row(board, player):
#     """
#     Detect ANY 4-in-a-row threat for `player`.
#     Returns True if the player has a pattern that can win in 1 move.
#     Detects:
#       - open four: .XXXX.
#       - closed four: XXXX.
#       - closed four: .XXXX
#       - broken fours: X.XXX, XX.XX, XXX.X
#     """

#     size = board.shape[0]
#     target = player

#     # Directions: (dr, dc)
#     directions = [
#         (0, 1),   # horizontal
#         (1, 0),   # vertical
#         (1, 1),   # diag \
#         (1, -1),  # diag /
#     ]

#     for r in range(size):
#         for c in range(size):
#             if board[r, c] != target:
#                 continue

#             for dr, dc in directions:
#                 stones = []

#                 # Collect 6 cells centered around (r,c)
#                 for k in range(-3, 3):
#                     rr = r + dr * k
#                     cc = c + dc * k
#                     if 0 <= rr < size and 0 <= cc < size:
#                         stones.append(board[rr, cc])
#                     else:
#                         stones.append(None)  # out of bounds

#                 # Convert to simple list of values
#                 # Replace None with a blocker value
#                 line = [1 if s == target else 0 if s == 0 else -1 for s in stones]

#                 # Sliding window of length 5 or 6
#                 for L in (5, 6):
#                     for i in range(len(line) - L + 1):
#                         window = line[i:i+L]

#                         # Count stones and empties
#                         stones_count = window.count(1)
#                         empty_count = window.count(0)

#                         # Threat if exactly 4 stones and 1 empty
#                         if stones_count == 4 and empty_count >= 1:
#                             return True

#     return False

def detect_four_in_a_row_bitboard(board: np.ndarray, player: int, win_len: int = 5) -> bool:
    """
    Bitboard-style detection of any 'four in a row' for `player`.
    Definition: in some length-`win_len` window, there are exactly 4 stones
    of `player`, at least 1 empty, and no opponent stones.

    Works for arbitrary board size (<= 19 comfortably).
    """
    size = board.shape[0]
    opponent = WHITE if player == BLACK else BLACK

    # Helper: given a 1D line (np.array), build two bitboards:
    #   bb_player: bits where player has stones
    #   bb_opp:    bits where opponent has stones
    def line_bitboards(line: np.ndarray):
        bb_p = 0
        bb_o = 0
        for i, v in enumerate(line):
            if v == player:
                bb_p |= (1 << i)
            elif v == opponent:
                bb_o |= (1 << i)
        return bb_p, bb_o, len(line)

    # Helper: scan a bitboard line for any 4-in-win_len window
    def has_four_in_bitline(bb_p: int, bb_o: int, length: int) -> bool:
        if length < win_len:
            return False
        # Precompute a sliding mask of length win_len
        mask = (1 << win_len) - 1
        for start in range(length - win_len + 1):
            window_mask = mask << start
            # If opponent has any stone in window, skip
            if bb_o & window_mask:
                continue
            # Count player stones in window
            stones = bb_p & window_mask
            count_player = stones.bit_count()
            if count_player == win_len - 1:
                # At least one empty is guaranteed because opponent has none
                return True
        return False

    # Rows
    for r in range(size):
        line = board[r, :]
        bb_p, bb_o, length = line_bitboards(line)
        if has_four_in_bitline(bb_p, bb_o, length):
            return True

    # Columns
    for c in range(size):
        line = board[:, c]
        bb_p, bb_o, length = line_bitboards(line)
        if has_four_in_bitline(bb_p, bb_o, length):
            return True

    # Diagonals (\)
    for offset in range(-size + win_len, size - win_len + 1):
        diag = np.diag(board, k=offset)
        bb_p, bb_o, length = line_bitboards(diag)
        if has_four_in_bitline(bb_p, bb_o, length):
            return True

    # Anti-diagonals (/)
    flipped = np.fliplr(board)
    for offset in range(-size + win_len, size - win_len + 1):
        diag = np.diag(flipped, k=offset)
        bb_p, bb_o, length = line_bitboards(diag)
        if has_four_in_bitline(bb_p, bb_o, length):
            return True

    return False

def count_open_threes_in_bitline(bb_p: int, bb_o: int, length: int, win_len: int = 5) -> int:
    """
    Count open-three patterns in a bitboard line.
    An open-three is a window of length win_len (usually 5) with:
      - exactly 3 stones of player
      - at least 2 empties
      - no opponent stones
      - and at least one way to extend to an open-four
    """
    if length < win_len:
        return 0

    mask = (1 << win_len) - 1
    count = 0

    for start in range(length - win_len + 1):
        window_mask = mask << start

        # Opponent stone inside window → not a threat
        if bb_o & window_mask:
            continue

        stones = bb_p & window_mask
        num_player = stones.bit_count()
        num_empty = win_len - num_player

        if num_player == 3 and num_empty >= 2:
            # Check if this window can become an open-four
            # i.e., at least one empty cell is adjacent to the 3 stones
            # and not blocked by opponent
            # (simple heuristic: ensure window edges are empty)
            left_edge = start - 1
            right_edge = start + win_len

            left_ok = (left_edge >= 0)
            right_ok = (right_edge < length)

            # If either side is empty, it's an open-three
            if (left_ok and not (bb_o & (1 << left_edge))) or \
               (right_ok and not (bb_o & (1 << right_edge))):
                count += 1

    return count

def detect_double_three_bitboard(board: np.ndarray, player: int, win_len: int = 5) -> bool:
    """
    Detects whether the board contains a DOUBLE-THREE for `player`.
    That is: two or more open-three threats in any directions.
    """
    size = board.shape[0]
    opponent = WHITE if player == BLACK else BLACK

    def line_bitboards(line):
        bb_p = 0
        bb_o = 0
        for i, v in enumerate(line):
            if v == player:
                bb_p |= (1 << i)
            elif v == opponent:
                bb_o |= (1 << i)
        return bb_p, bb_o, len(line)

    total_threes = 0

    # Rows
    for r in range(size):
        bb_p, bb_o, length = line_bitboards(board[r, :])
        total_threes += count_open_threes_in_bitline(bb_p, bb_o, length, win_len)
        if total_threes >= 2:
            return True

    # Columns
    for c in range(size):
        bb_p, bb_o, length = line_bitboards(board[:, c])
        total_threes += count_open_threes_in_bitline(bb_p, bb_o, length, win_len)
        if total_threes >= 2:
            return True

    # Diagonals (\)
    for offset in range(-size + win_len, size - win_len + 1):
        diag = np.diag(board, k=offset)
        bb_p, bb_o, length = line_bitboards(diag)
        total_threes += count_open_threes_in_bitline(bb_p, bb_o, length, win_len)
        if total_threes >= 2:
            return True

    # Anti-diagonals (/)
    flipped = np.fliplr(board)
    for offset in range(-size + win_len, size - win_len + 1):
        diag = np.diag(flipped, k=offset)
        bb_p, bb_o, length = line_bitboards(diag)
        total_threes += count_open_threes_in_bitline(bb_p, bb_o, length, win_len)
        if total_threes >= 2:
            return True

    return False

def opponent_winning_moves(env, opponent: int):
    """
    Return list of moves where `opponent` wins immediately.
    """
    winning = []
    legal = env.get_valid_moves()
    for a in legal:
        tmp = env.clone()
        tmp.current_player = opponent
        tmp.make_move(a)
        res = tmp.check_result()
        if res == (GameResult.BLACK_WIN if opponent == BLACK else GameResult.WHITE_WIN):
            winning.append(a)
    return winning

def winning_moves(env, player: int):
    """All moves where `player` wins immediately."""
    wins = []
    legal = env.get_valid_moves()
    for a in legal:
        tmp = env.clone()
        tmp.current_player = player
        tmp.make_move(a)
        res = tmp.check_result()
        if res == (GameResult.BLACK_WIN if player == BLACK else GameResult.WHITE_WIN):
            wins.append(a)
    return wins

# def classify_threat(env, a: int, player: int, win_len: int = 5) -> str:
#     """
#     Classify move `a` for `player` into a DEE-lite threat category.
#     """
#     tmp = env.clone()
#     tmp.make_move(a)

#     # 1) Immediate win?
#     res = tmp.check_result()
#     if res == (GameResult.BLACK_WIN if player == BLACK else GameResult.WHITE_WIN):
#         return "win"

#     # 2) How many immediate wins next turn (double-four / VCF-lite)?
#     #    Assume opponent moves, then we move again.
#     #    Here we approximate by: from tmp, count our winning moves.
#     my_next_wins = winning_moves(tmp, player)
#     if len(my_next_wins) >= 2:
#         return "double_four"

#     # 3) Any four-in-a-row (bitboard-based)?
#     if detect_four_in_a_row_bitboard(tmp.board, player, win_len=win_len):
#         return "four"

#     return "other"

def classify_threat(env, a: int, player: int, win_len: int = 5) -> str:
    """
    Classify move `a` for `player` into a DEE-lite threat category.

    Categories:
      - "win"          : immediate 5-in-a-row
      - "block"        : prevents opponent's immediate win
      - "double_four"  : creates >=2 immediate winning replies next turn
      - "four"         : creates any 4-in-a-row threat
      - "other"        : no special tactical significance
    """

    opponent = WHITE if player == BLACK else BLACK

    # --- 0) Defensive block: does this move stop opponent's immediate win? ---
    opp_wins_now = winning_moves(env, opponent)
    if a in opp_wins_now:
        return "block"

    # --- Simulate the move ---
    tmp = env.clone()
    tmp.make_move(a)

    # --- 1) Immediate win for player? ---
    res = tmp.check_result()
    if res == (GameResult.BLACK_WIN if player == BLACK else GameResult.WHITE_WIN):
        return "win"

    # --- 2) Double-four: creates >=2 immediate wins next turn ---
    my_next_wins = winning_moves(tmp, player)
    if len(my_next_wins) >= 2:
        return "double_four"

    # --- 3) Single four: any 4-in-a-row threat (bitboard-based) ---
    if detect_four_in_a_row_bitboard(tmp.board, player, win_len=win_len):
        return "four"

    # --- 4) Nothing special ---
    return "other"