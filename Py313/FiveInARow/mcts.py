import math
import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, Tuple, Dict
from collections import defaultdict
from datetime import datetime

from utils import (
    GameResult,
    BLACK,
    WHITE,
    BOARD_SIZE,
    # opponent_winning_moves,
    # creates_open_four,
    # creates_double_three,
    # detect_four_in_a_row_bitboard,
    winning_moves,
    classify_threat,
    detect_double_three_bitboard,
)
from gomoku import Gomoku

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'] = None,
                 action_from_parent: Optional[int] = None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.P: Dict[int, float] = {}
        self.N: Dict[int, int] = defaultdict(int)
        self.W: Dict[int, float] = defaultdict(float)
        self.Q: Dict[int, float] = defaultdict(float)
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_expanded: bool = False
        self.current_player: Optional[int] = None
        self.value_eval: Optional[float] = None

    def add_dirichlet_noise(self, alpha, epsilon):
        actions = list(self.P.keys())

        if len(actions) == 0:
            print("[Dirichlet] No actions, skipping noise")
            return

        # Force scalar alpha/epsilon
        alpha = float(alpha)
        epsilon = float(epsilon)

        try:
            actions = [int(a) for a in actions]
        except Exception as e:
            print("[Dirichlet] ERROR converting actions to int:", actions)
            raise

        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.P[a] = (1 - epsilon) * self.P[a] + epsilon * float(n)

# -----------------------------
# MCTS Core
# -----------------------------
class MCTS:
    """
    AlphaZero-style MCTS.

    - Uses policy/value nets to expand/evaluate nodes
    - Builds π from root visit counts
    - Supports focused legal moves and optional pruning
    """

    def __init__(
        self,
        env_cls,
        encode_state_fn,
        policy_net,
        value_net,
        sims_per_move: int = 200,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        self.env_cls = env_cls
        self.encode_state = encode_state_fn
        self.policy_net = policy_net
        self.value_net = value_net

        self.sims_per_move = sims_per_move
        self.c_puct = c_puct
        self.temperature = temperature

        # Ensure these are plain floats (not tuples)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)

        self.average_q_at_root = 0.0
        self.limit_rollout_depth = 9 # 5  # max depth before cutoff

    # ---------------------------------------------------------
    # Focused legal moves (radius around existing stones)
    # ---------------------------------------------------------
    def _focused_legal(self, env: Gomoku, radius: int = 6):
        legal = env.get_valid_moves()
        if env.moves_played == 0:
            return [int(a) for a in legal]

        stones = np.argwhere(env.board != 0)
        if stones.size == 0:
            return [int(a) for a in legal]

        allowed = []
        size = env.size
        for a in legal:
            r, c = divmod(int(a), size)
            dmin = np.min(np.abs(stones[:, 0] - r) + np.abs(stones[:, 1] - c))
            if dmin <= radius:
                allowed.append(int(a))

        return allowed if allowed else [int(a) for a in legal]

    # ---------------------------------------------------------
    # Selection: traverse tree using PUCT, return path + leaf
    # ---------------------------------------------------------
    def _select(self, node: MCTSNode, env: Gomoku):
        """
        Traverse from `node` downwards using PUCT until:
          - we hit an unexpanded node, or
          - we reach a terminal state.

        Returns:
            path: list of (node, action)
            leaf: final node reached
            leaf_env: environment at leaf
            parent_action: action from leaf.parent to leaf (or -1 at root)
        """
        path = []

        if not node.is_expanded:
            return path, node, env, -1

        while True:
            if env.check_result() != GameResult.ONGOING:
                return path, node, env, -1

            legal_int = self._focused_legal(env, radius=2)

            total_N = sum(node.N.get(a, 0) for a in legal_int) + 1e-8
            best_action, best_score = None, -float("inf")

            # Compute average Q for FPU (First Play Urgency)
            avg_q = np.mean([node.Q[a] for a in node.Q.keys()]) if node.Q else 0.0
            fpu = avg_q - 0.1  # can be tuned

            for a in legal_int:
                if node.N.get(a, 0) == 0:
                    Q = node.Q.get(a, fpu)
                else:
                    Q = node.Q[a]

                P = node.P.get(a, 0.0)
                U = self.c_puct * P * math.sqrt(total_N) / (1 + node.N.get(a, 0))
                s = Q + U

                if s > best_score:
                    best_score, best_action = s, a

            a = int(best_action)
            env.make_move(a)
            path.append((node, a))

            if a in node.children:
                node = node.children[a]
                if not node.is_expanded:
                    return path, node, env, a
            else:
                child = MCTSNode(parent=node, action_from_parent=a)
                node.children[a] = child
                return path, child, env, a

    # ---------------------------------------------------------
    # Backup along path
    # ---------------------------------------------------------
    def _backup_path(
        self,
        path,
        leaf_value,
        leaf_env_current_player,
        path_first_node_player,
    ):
        """
        Backup value from leaf to root.

        `leaf_value` is from the perspective of `leaf_env_current_player`.
        We flip signs along the path to alternate players.
        """
        if leaf_value is None:
            return

        # Align leaf value to the perspective of the first node in the path
        v = leaf_value if leaf_env_current_player == path_first_node_player else -leaf_value

        flip = 1
        for node, a in reversed(path):
            node.N[a] = node.N.get(a, 0) + 1
            node.W[a] = node.W.get(a, 0.0) + (v if flip == 1 else -v)
            node.Q[a] = node.W[a] / node.N[a]
            flip *= -1

    # ---------------------------------------------------------
    # Expansion + evaluation
    # ---------------------------------------------------------
    def _expand_and_evaluate(self, node: MCTSNode, env: Gomoku, parent_action: int):
        """
        Expand a leaf node:
          - get focused legal moves
          - (optionally) prune with DEE-style rules
          - query policy/value nets
          - set priors and children

        Returns:
            child node corresponding to `parent_action` if parent_action != -1,
            otherwise None.
        """
        legal_int = self._focused_legal(env, radius=2)
        legal_int = dee_prune_moves(
            env,
            legal_int,
            win_len=5,
            # check_open_four=False,
            # check_double_three=False,
        )

        # Safety fallback
        if len(legal_int) < 3:
            legal_int = self._focused_legal(env, radius=2)


        state_t = self.encode_state(env.board, env.current_player)

        with torch.no_grad():
            logits = self.policy_net(state_t)
            priors_all = F.softmax(logits, dim=-1).squeeze(0)
            value = float(self.value_net(state_t).item())

        # Cache NN value on this node
        node.value_eval = value

        priors_np = priors_all.detach().cpu().numpy()
        masked_priors = np.zeros_like(priors_np, dtype=np.float32)

        if len(legal_int) > 0:
            masked_priors[legal_int] = priors_np[legal_int]
            s = masked_priors.sum()
            if s > 0:
                masked_priors /= s
            else:
                masked_priors[legal_int] = 1.0 / len(legal_int)

        if not node.is_expanded:
            node.current_player = env.current_player
            for a in legal_int:
                node.P[a] = float(masked_priors[a])
                node.N[a] = 0
                node.W[a] = 0.0
                node.Q[a] = 0.0
            node.is_expanded = True

        # Create children lazily
        for a in legal_int:
            if a not in node.children:
                child = MCTSNode(parent=node, action_from_parent=a)
                child.current_player = -env.current_player
                node.children[a] = child

        if parent_action != -1:
            a = int(parent_action)
            if a not in node.children:
                child = MCTSNode(parent=node, action_from_parent=a)
                child.current_player = -env.current_player
                node.children[a] = child
            return node.children[a]

        return None
    
    # ---- Simulation loop ----
    def run(
        self,
        root_env: Gomoku,
        root: Optional[MCTSNode] = None,
        add_root_noise: bool = True,
    ) -> Tuple[MCTSNode, np.ndarray]:
        """
        Run MCTS simulations starting from root_env state.

        Parameters:
        - root_env: the current Gomoku environment
        - root: optional existing root node for tree reuse
        - add_root_noise: whether to apply Dirichlet noise at the root

        Returns:
        - root node (possibly reused and expanded)
        - policy target pi: visit-count distribution over actions
        """

        cutoff_hits = 0
        cutoff_values = []

        # --- ROOT REUSE LOGIC ---
        if root is None:
            # No previous tree → create a fresh root
            root = MCTSNode(parent=None)
        else:
            # Reusing previous subtree → detach from old parent
            root.parent = None

        self.root = root

        # --- ROOT EXPANSION / PRIORS ---
        if not getattr(root, "is_expanded", False):
            # Fresh root (or reused but not yet expanded): expand using current env
            legal_int = self._focused_legal(root_env, radius=2)

            state_t = self.encode_state(root_env.board, root_env.current_player)
            with torch.no_grad():
                root_logits = self.policy_net(state_t)
                priors_all = F.softmax(root_logits, dim=-1).squeeze(0)

            priors_np = priors_all.detach().cpu().numpy()
            masked_priors = np.zeros_like(priors_np, dtype=np.float32)
            if len(legal_int) > 0:
                masked_priors[legal_int] = priors_np[legal_int]
                s = masked_priors.sum()
                if s > 0:
                    masked_priors /= s
                else:
                    masked_priors[legal_int] = 1.0 / len(legal_int)

            root.current_player = root_env.current_player
            for a in legal_int:
                root.P[a] = float(masked_priors[a])
                root.N[a] = 0
                root.W[a] = 0.0
                root.Q[a] = 0.0
            root.is_expanded = True
        else:
            # Reused, already-expanded root: legal moves are its existing priors
            legal_int = list(root.P.keys())

        # --- DIRICHLET NOISE (ONLY AFTER EXPANSION) ---
        if add_root_noise and len(legal_int) > 0:
            # Ensure scalar floats (in case caller passed tuples)
            alpha = float(self.dirichlet_alpha)
            eps = float(self.dirichlet_eps)
            root.add_dirichlet_noise(alpha=alpha, epsilon=eps)

        # --- Run simulations ---
        for _ in range(self.sims_per_move):
            sim_env = self._clone_env(root_env)

            # SELECT
            path, leaf, leaf_env, parent_action = self._select(root, sim_env)
            depth = len(path)

            # TERMINAL?
            if leaf_env.check_result() != GameResult.ONGOING:
                if leaf_env.find_any_five() == (None, []):
                    print("[MCTS] Warning: Terminal state reached but no winner found.")

                v_eval = self._terminal_value(leaf_env, leaf.current_player)
                self._backup_path(
                    path,
                    v_eval,
                    leaf_env.current_player,
                    path[0][0].current_player if path else root.current_player,
                )
                continue

            # EARLY CUTOFF
            if depth >= self.limit_rollout_depth:
                state_t = self.encode_state(leaf_env.board, leaf_env.current_player)
                with torch.no_grad():
                    v_eval = float(self.value_net(state_t).item())
                self._backup_path(
                    path,
                    v_eval,
                    leaf_env.current_player,
                    path[0][0].current_player if path else root.current_player,
                )
                cutoff_hits += 1
                cutoff_values.append(v_eval)
                continue

            # EXPAND + EVAL
            child = self._expand_and_evaluate(leaf, leaf_env, parent_action)
            v_eval = child.value_eval if child else leaf.value_eval
            self._backup_path(
                path,
                v_eval,
                leaf_env.current_player,
                path[0][0].current_player if path else root.current_player,
            )

        # --- Build policy target ---
        pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        total_visits = sum(root.N.get(a, 0) for a in legal_int)

        if total_visits == 0:
            # Fall back to priors if no visits (should be rare)
            for a in legal_int:
                pi[a] = root.P.get(a, 0.0)
        else:
            temp = max(self.temperature, 1e-6)
            for a in legal_int:
                pi[a] = (root.N.get(a, 0) ** (1.0 / temp))

        s = pi[legal_int].sum()
        if s > 0:
            pi[legal_int] /= s
        else:
            if len(legal_int) > 0:
                pi[legal_int] = 1.0 / len(legal_int)

        # Logging
        if cutoff_hits > 0:
            avg_cutoff_val = sum(cutoff_values) / cutoff_hits
            print(
                f"[MCTS] Cutoff hits this move: {cutoff_hits}/{self.sims_per_move}, "
                f"avg value_net prediction={avg_cutoff_val:.4f}"
            )
        else:
            print(f"[MCTS] Cutoff hits this move: 0/{self.sims_per_move}")

        if len(legal_int) > 0:
            visited_actions = [a for a in legal_int if root.N.get(a, 0) > 0]
            avg_q = float(np.mean([root.Q[a] for a in visited_actions])) if visited_actions else 0.0
            print(f"[MCTS] Average Q at root: {avg_q:.4f}")
            self.average_q_at_root = avg_q

        return root, pi
    
    # ---- Utilities ----
    def _clone_env(self, env: Gomoku) -> Gomoku:
        """Create a deep copy of the environment state."""
        new_env = self.env_cls(size=env.size, win_len=env.win_len)
        new_env.board = env.board.copy()
        new_env.current_player = env.current_player
        new_env.moves_played = env.moves_played
        new_env.last_move = env.last_move
        return new_env

    def _terminal_value(self, env: Gomoku, reference_player: int) -> float:
        """
        Return terminal value from the perspective of reference_player.
        +1 if reference_player wins, -1 if loses, 0 for draw.
        """
        res = env.check_result()
        if res == GameResult.DRAW:
            return 0.0
        winner = BLACK if res == GameResult.BLACK_WIN else WHITE
        return 1.0 if winner == reference_player else -1.0


# ---------------------------------------------------------
# DEE-style pruning (corrected)
# ---------------------------------------------------------
def dee_prune_moves(
    env,
    legal_moves,
    win_len: int = 5,
):
    """
    DEE-lite threat hierarchy for Gomoku.

    Priority:
      1) Immediate wins for current player
      2) Immediate blocks against opponent's wins
      3) Double-fours (creates >=2 winning replies)
      4) Double-threes (creates >=2 open-three threats)
      5) Single fours
      6) Otherwise: return all legal_moves (no starvation)

    Returns a prioritized move list.
    """

    player = env.current_player
    opponent = WHITE if player == BLACK else BLACK

    legal_moves = list(legal_moves)

    # --- 1) Immediate wins ---
    my_wins = winning_moves(env, player)
    if my_wins:
        return my_wins

    # --- 2) Immediate blocks (opponent wins now) ---
    opp_wins = winning_moves(env, opponent)
    if opp_wins:
        # Prioritize blocks but keep full move set
        return list(dict.fromkeys(opp_wins + legal_moves))

    # --- 3) Classify remaining moves ---
    double_fours = []
    double_threes = []
    fours = []

    for a in legal_moves:
        # Use classify_threat for offensive categories
        t = classify_threat(env, a, player, win_len=win_len)

        if t == "double_four":
            double_fours.append(a)
            continue

        if t == "four":
            fours.append(a)
            continue

        # --- Double-three detection (bitboard-based) ---
        tmp = env.clone()
        tmp.make_move(a)
        if detect_double_three_bitboard(tmp.board, player, win_len=win_len):
            double_threes.append(a)
            continue

    # --- 3a) Double-fours: strongest non-immediate threats ---
    if double_fours:
        return list(dict.fromkeys(double_fours + legal_moves))

    # --- 3b) Double-threes: next strongest forcing threats ---
    if double_threes:
        return list(dict.fromkeys(double_threes + legal_moves))

    # --- 3c) Single fours ---
    if fours:
        return list(dict.fromkeys(fours + legal_moves))

    # --- 4) No special threats: return all ---
    return legal_moves