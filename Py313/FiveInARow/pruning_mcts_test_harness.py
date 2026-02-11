import numpy as np

from gomoku import Gomoku
from utils import BLACK, WHITE, GameResult
from mcts import dee_prune_moves, winning_moves, classify_threat
from utils import detect_four_in_a_row_bitboard


def make_test_env():
    """
    Create a Gomoku environment with a forced-block situation:
    Black has XXXX_ and White must block.
    """
    env = Gomoku(size=15, win_len=5)
    env.reset()

    # Build XXXX_
    # Row 7, columns 7-10 are black, col 11 is empty
    r = 7
    env.board[r, 7]  = BLACK
    env.board[r, 8]  = BLACK
    env.board[r, 9]  = BLACK
    env.board[r, 10] = BLACK
    # env.board[r, 11] = 0  # empty

    env.current_player = WHITE  # White to move, must block
    env.moves_played = 4
    env.last_move = (r, 10)

    return env


def test_pruning():
    env = make_test_env()
    legal = env.get_valid_moves()

    print("=== TEST POSITION ===")
    print("White must block Black's XXXX_ at (7,11)")
    print()

    # 1. Detect opponent immediate wins
    opp_wins = winning_moves(env, BLACK)
    print("Black immediate winning moves:", opp_wins)

    # 2. Detect four-in-a-row threat
    four_threat = detect_four_in_a_row_bitboard(env.board, BLACK)
    print("Black has four-in-a-row threat:", four_threat)

    # 3. Classify the block move
    block_move = 7 * env.size + 11
    cls = classify_threat(env, block_move, WHITE)
    print(f"White's block move classify_threat: {cls}")

    # 4. Run DEE pruning
    pruned = dee_prune_moves(env, legal, win_len=5)
    print("DEE-pruned move list (first 10):", pruned[:10])

    # 5. Check if block is first
    # if pruned[0] == block_move:
    #     print("✔ PASS: Block move is first in pruned list")
    # else:
    #     print("✘ FAIL: Block move is NOT first")
    #     print("Expected:", block_move, "Got:", pruned[0])
    expected_blocks = set(opp_wins)
    actual_prefix = set(pruned[:len(expected_blocks)])

    if actual_prefix == expected_blocks:
        print("✔ PASS: All forced blocks appear first in pruned list")
    else:
        print("✘ FAIL: Forced blocks not correctly prioritized")
        print("Expected prefix:", expected_blocks)
        print("Actual prefix:", actual_prefix)    

    # 6. Check if block is included at all
    if block_move in pruned:
        print("✔ PASS: Block move is included in pruned list")
    else:
        print("✘ FAIL: Block move missing from pruned list")


if __name__ == "__main__":
    test_pruning()