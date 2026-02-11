"""
Gomoku DRL Skeleton (PyTorch)
- Environment: Gomoku (15x15, two players, first to get exactly five in a row wins)
- Agent: DRL with policy/value networks, experience replay
- GUI: Optional (tkinter), interactive play
- Main: Player vs NN, NN vs NN. Training (backprop) after each game.
"""

import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch

from gomoku import Gomoku
from gui import GomokuGUI
from drl import DRL
from mcts import MCTS
from utils import GameResult, Transition, coord_to_index, BLACK, WHITE


# -----------------------------
# Helpers
# -----------------------------
def random_first_move(board_size: int = 15) -> Tuple[int, int]:
    row = random.randint(0, board_size - 1)
    col = random.randint(0, board_size - 1)
    return row, col


# -----------------------------
# Main game loop
# -----------------------------
def play_game(
    env: Gomoku,
    agent: DRL,
    mode: str = "pve",
    human_is_black: bool = True,
    use_mcts: bool = False,
    gui: Optional[GomokuGUI] = None,
    deterministic_vs_human: bool = True,
) -> Tuple[GameResult, Dict[str, float]]:
    """
    Play a single game of Gomoku.

    Modes:
      - "pve": Player vs NN (optionally with MCTS)
      - "eve": NN vs NN
      - "pvmcts": Player vs pure MCTS
      - "nn_vs_mcts": Deterministic NN vs MCTS
      - "pve_policy": Player vs raw policy_net
      - "pve_value": Player vs raw value_net
    """

    # Single MCTS instance per game (tree reuse via `root`)
    mcts = MCTS(
        Gomoku,
        agent.encode_state,
        agent.policy,
        agent.value,
        sims_per_move=400, # 400,
        c_puct=2.5,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    )
    root = None

    env.reset()
    transitions: List[Transition] = []

    while True:
        legal = env.get_valid_moves()
        result = env.check_result()
        if result != GameResult.ONGOING:
            break

        # -------------------------
        # Human turn?
        # -------------------------
        is_human_turn = (
            (mode.startswith("pve") or mode == "pvmcts")
            and (
                (env.current_player == BLACK and human_is_black)
                or (env.current_player == WHITE and not human_is_black)
            )
        )

        if is_human_turn:
            prev_board = env.board.copy()
            prev_player = env.current_player

            if gui:
                gui.draw_stones()
                action = None
                while action is None:
                    gui.window.update()
                    action = gui.get_user_action()
                if action not in legal or not env.make_move(action):
                    continue
            else:
                env.render()
                print("Enter move as 'row col' (0-based):")
                try:
                    r, c = map(int, input().strip().split())
                    action = coord_to_index(r, c, env.size)
                except Exception:
                    print("Invalid input. Try again.")
                    continue
                if action not in legal or not env.make_move(action):
                    print("Illegal move. Try again.")
                    continue

            # env.current_player has already toggled, so previous player is:
            played_by = WHITE if env.current_player == BLACK else BLACK

            transitions.append(
                Transition(
                    state=prev_board,
                    action=action,
                    reward=0.0,
                    next_state=env.board.copy(),
                    done=False,
                    player=played_by,
                    q_value=0.0,
                    pi=None,
                    z=None,
                )
            )

        else:
            prev_board = env.board.copy()
            prev_player = env.current_player

            # -------------------------
            # Branch by mode
            # -------------------------
            if mode in ("pvmcts", "nn_vs_mcts"):
                # Pure MCTS side
                root, pi = mcts.run(env, root=root)

                pi_legal = np.array([pi[a] for a in legal], dtype=np.float32)
                s = pi_legal.sum()
                if s > 0:
                    pi_legal /= s
                else:
                    pi_legal = np.ones(len(legal), dtype=np.float32) / len(legal)

                action = np.random.choice(legal, p=pi_legal)
                env.make_move(action)

                # Tree reuse
                if root is not None and action in root.children:
                    root = root.children[action]
                    root.parent = None
                else:
                    root = None

                transitions.append(
                    Transition(
                        state=prev_board,
                        action=action,
                        reward=0.0,
                        next_state=env.board.copy(),
                        done=False,
                        player=prev_player,
                        q_value=getattr(mcts, "average_q_at_root", 0.0),
                        pi=pi.copy(),
                        z=None,
                    )
                )

            elif mode == "pve_policy":
                # Direct argmax from policy_net
                state_t = agent.encode_state(env.board, env.current_player)
                with torch.no_grad():
                    logits = agent.policy(state_t).cpu().numpy().flatten()
                mask = np.full_like(logits, -np.inf)
                mask[legal] = logits[legal]
                action = int(np.argmax(mask))
                env.make_move(action)

                transitions.append(
                    Transition(
                        state=prev_board,
                        action=action,
                        reward=0.0,
                        next_state=env.board.copy(),
                        done=False,
                        player=prev_player,
                        q_value=0.0,
                        pi=None,
                        z=None,
                    )
                )

            elif mode == "pve_value":
                # Pick move maximizing value_net prediction
                best_action, best_val = None, -float("inf")
                for a in legal:
                    tmp_env = env.clone()
                    tmp_env.make_move(a)
                    state_t = agent.encode_state(tmp_env.board, tmp_env.current_player)
                    with torch.no_grad():
                        val = float(agent.value(state_t).item())
                    if val > best_val:
                        best_val, best_action = val, a

                action = best_action
                env.make_move(action)

                transitions.append(
                    Transition(
                        state=prev_board,
                        action=action,
                        reward=0.0,
                        next_state=env.board.copy(),
                        done=False,
                        player=prev_player,
                        q_value=best_val,
                        pi=None,
                        z=None,
                    )
                )

            else:
                # Default NN branch (pve/eve with optional MCTS)
                if use_mcts:
                    if env.moves_played == 0:
                        # Random first stone (no search)
                        board_size = env.size
                        row = random.randint(0, board_size - 1)
                        col = random.randint(0, board_size - 1)
                        action = row * board_size + col
                        env.make_move(action)

                        pi = np.zeros(board_size * board_size, dtype=np.float32)
                        pi[action] = 1.0

                        transitions.append(
                            Transition(
                                state=prev_board,
                                action=action,
                                reward=0.0,
                                next_state=env.board.copy(),
                                done=False,
                                player=prev_player,
                                q_value=0.0,
                                pi=pi,
                                z=None,
                            )
                        )
                    else:
                        # Normal MCTS branch
                        root, pi = mcts.run(env, root=root)

                        pi_legal = np.array([pi[a] for a in legal], dtype=np.float32)
                        s = pi_legal.sum()
                        if s <= 0:
                            pi_legal = np.ones(len(legal), dtype=np.float32) / len(legal)
                        else:
                            pi_legal /= s

                        entropy = -np.sum(pi_legal * np.log(pi_legal + 1e-12))
                        nonzero_count = np.count_nonzero(pi_legal > 0)
                        print(
                            f"[PlayGame] π_legal entropy={entropy:.4f}, "
                            f"non‑zero entries={nonzero_count}/{len(pi_legal)}"
                        )

                        if mode == "pve" and deterministic_vs_human:
                            action = legal[np.argmax(pi_legal)]
                        else:
                            action = np.random.choice(legal, p=pi_legal)

                        board_size = env.size
                        r_sel, c_sel = divmod(action, board_size)
                        print(
                            f"[PlayGame] Selected move: ({r_sel},{c_sel}) -> action={action}"
                        )

                        env.make_move(action)

                        # Tree reuse
                        if root is not None and action in root.children:
                            root = root.children[action]
                            root.parent = None
                        else:
                            root = None

                        avg_q = getattr(mcts, "average_q_at_root", 0.0)

                        transitions.append(
                            Transition(
                                state=prev_board,
                                action=action,
                                reward=0.0,
                                next_state=env.board.copy(),
                                done=False,
                                player=prev_player,
                                q_value=avg_q,
                                pi=pi.copy(),
                                z=None,
                            )
                        )
                else:
                    # Pure NN (epsilon-greedy in DRL.select_action)
                    action = agent.select_action(env, legal)
                    env.make_move(action)

                    transitions.append(
                        Transition(
                            state=prev_board,
                            action=action,
                            reward=0.0,
                            next_state=env.board.copy(),
                            done=False,
                            player=prev_player,
                            q_value=0.0,
                            pi=None,
                            z=None,
                        )
                    )

        if gui:
            gui.draw_stones()

    # -----------------------------
    # Game ended
    # -----------------------------
    if gui:
        gui.draw_stones()
    else:
        env.render()
    print(f"Result: {result.name}")

    # Mark transitions done and assign rewards
    for t in transitions:
        t.done = True
    DRL.compute_rewards(result, transitions)

    # Assign z for AlphaZero-style training
    if result == GameResult.DRAW:
        z_map = {BLACK: 0.0, WHITE: 0.0}
    else:
        winner = BLACK if result == GameResult.BLACK_WIN else WHITE
        z_map = {winner: 1.0, -winner: -1.0}

    for t in transitions:
        t.z = z_map[t.player]

    # Store transitions
    for t in transitions:
        agent.store_transition(t)

    # Choose training style
    if use_mcts:
        stats = agent.train_after_game_az()
    else:
        stats = agent.train_after_game()

    return result, stats