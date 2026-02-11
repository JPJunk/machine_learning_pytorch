"""
Gomoku DRL Skeleton (PyTorch)
- Environment: Gomoku (15x15, two players, first to get exactly five in a row wins)
- Agent: DRL with policy/value network stubs, experience replay
- GUI: Optional stub (tkinter), interactive PVE
- Main: Player vs NN, NN vs NN. Training (backprop) after each game.
"""

from datetime import datetime

import numpy as np
import torch

from gomoku import Gomoku
from gui import GomokuGUI
from drl import DRL
from utils import GameResult
from agent_persistence import AgentPersistence
from play_game import play_game


def test_policy_net_outputs(policy_net, board_size=15, device="cpu"):
    """
    Feed an empty board into the policy net and print the full 15x15 logits grid.
    """
    policy_net.eval()
    state = np.zeros((2, board_size, board_size), dtype=np.float32)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = policy_net(state_t).squeeze(0).cpu().numpy()

    logits_grid = logits.reshape(board_size, board_size)

    print("Policy logits on empty board:")
    for r in range(board_size):
        row_str = " ".join(f"{logits_grid[r, c]:6.3f}" for c in range(board_size))
        print(row_str)


def test_value_net_outputs(value_net, board_size=15, num_samples=5, device="cpu"):
    """
    Sample random Gomoku boards and print value_net predictions.
    Also prints the board layout for each sample.
    """
    value_net.eval()
    for i in range(num_samples):
        board = np.random.choice(
            [0, 1, -1],
            size=(board_size, board_size),
            p=[0.8, 0.1, 0.1],
        )

        black = (board == 1).astype(np.float32)
        white = (board == -1).astype(np.float32)
        state = np.stack([black, white], axis=0)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            val = value_net(state_t).item()

        print(f"\nSample {i + 1}: value_net output = {val:.4f}")
        for r in range(board_size):
            row_str = " ".join(
                "X" if board[r, c] == 1 else "O" if board[r, c] == -1 else "."
                for c in range(board_size)
            )
            print(row_str)


def main():
    """
    Main entry point.
    - Choose mode: Player vs NN (pve) or NN vs NN (eve)
    - Player can choose Black or White
    - Toggle MCTS for NN moves
    - Backprop after each game
    - Results tracked and summarized
    - Agent state loaded at startup and saved after each game
    - GUI used for Player vs NN mode
    """
    env = Gomoku()
    agent = DRL()

    # Load previous agent state (method already handles exceptions)
    AgentPersistence.load(agent, "gomoku_agent.pth")

    # Optional sanity checks
    test_value_net_outputs(agent.value, board_size=env.size, num_samples=10, device=agent.device)
    test_policy_net_outputs(agent.policy, board_size=env.size, device=agent.device)

    print("Choose mode:")
    print("1) Player vs NN")
    print("2) NN vs NN")
    print("3) Player vs MCTS")
    print("4) NN (deterministic) vs MCTS")
    print("5) Player vs policy_net only")
    print("6) Player vs value_net only")

    choice = input("Enter number: ").strip()
    modes = {
        "1": "pve",
        "2": "eve",
        "3": "pvmcts",
        "4": "nn_vs_mcts",
        "5": "pve_policy",
        "6": "pve_value",
    }
    mode = modes.get(choice, "eve")

    if mode in ("pve", "eve"):
        use_mcts = input("Use MCTS for NN moves? (y/n): ").strip().lower() == "y"
    else:
        use_mcts = mode in ("pvmcts", "nn_vs_mcts")

    human_is_black = True
    if mode in ("pve", "pve_policy", "pve_value"):
        side = input("Play as Black(X) or White(O)? Enter B/W: ").strip().upper()
        human_is_black = (side == "B")

    try:
        num_games = int(input("How many games to play? (default 5): ").strip() or "5")
    except Exception:
        num_games = 5

    results = {
        GameResult.BLACK_WIN: 0,
        GameResult.WHITE_WIN: 0,
        GameResult.DRAW: 0,
    }

    gui = GomokuGUI(env) if mode in ("pve", "pvmcts", "pve_policy", "pve_value") else None

    for g in range(1, num_games + 1):
        print(f"\n=== Game {g} ===")
        res, stats = play_game(
            env,
            agent,
            mode=mode,
            human_is_black=human_is_black,
            use_mcts=use_mcts,
            gui=gui,
            deterministic_vs_human=True,
        )

        results[res] += 1

        timestamp = datetime.now().isoformat(timespec="seconds")
        agent.game_counter = getattr(agent, "game_counter", 0) + 1

        log_line = (
            f"Game {agent.game_counter} [{timestamp}]: result={res.name}, "
            f"policy_loss={stats.get('policy_loss', 0.0):.4f}, "
            f"value_loss={stats.get('value_loss', 0.0):.4f}, "
            f"total_loss={stats.get('total_loss', 0.0):.4f}"
        )
        with open("training_log.txt", "a") as f:
            f.write(log_line + "\n")
        print(f"[Log] {log_line}")

        AgentPersistence.save(agent, "gomoku_agent.pth")

    print("\nSummary:")
    for k, v in results.items():
        print(f"{k.name}: {v}")

    if use_mcts:
        print("\nAlphaZero-style training was applied using MCTS visit counts.")
    else:
        print("\nClassic DRL replay buffer training was applied.")


if __name__ == "__main__":
    main()