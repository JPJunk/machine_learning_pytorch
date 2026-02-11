import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from gomoku import Gomoku
from utils import Transition, GameResult, BLACK, WHITE, DEVICE
from policy_and_value_nets import PolicyNet, ValueNet

import symmetry_utils
from replay_buffer import ReplayBuffer


# -----------------------------
# DRL Agent (Policy + Value nets)
# -----------------------------
class DRL:
    """
    Deep RL agent for Gomoku using AlphaZero-style policy and value networks.

    Responsibilities:
      - Encode board states for NN input
      - Select actions (epsilon-greedy or softmax)
      - Store transitions (with symmetry augmentation)
      - Train policy/value networks after each game
      - Maintain replay buffer
    """

    def __init__(self, board_size=15,
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.board_size = board_size

        # -----------------------------
        # Networks
        # -----------------------------
        self.policy = PolicyNet(board_size=board_size).to(self.device)
        self.value  = ValueNet(board_size=board_size).to(self.device)

        # -----------------------------
        # Centrality bias (optional)
        # -----------------------------
        bias = gomoku_centrality_bias_for_policy(board_size, scale=0.1)
        with torch.no_grad():
            self.policy.head[-1].bias.copy_(bias)

        bias = gomoku_centrality_bias_for_value(board_size, scale=0.05)
        with torch.no_grad():
            # head = [Conv2d, BN, ReLU, Flatten, Linear, ReLU, Linear, Tanh]
            # The Linear(board_size^2 → 64) is index 4
            self.value.head[4].bias.copy_(
                bias[:board_size * board_size].mean().expand(64)
            )

        # -----------------------------
        # Hyperparameters
        # -----------------------------
        self.temperature = 1.0
        self.lr = 5e-4
        self.gamma = 0.99

        # -----------------------------
        # Optimizers
        # -----------------------------
        self.policy_opt = optim.Adam(
            self.policy.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self.value_opt = optim.Adam(
            self.value.parameters(), lr=self.lr, weight_decay=1e-4
        )

        # -----------------------------
        # Replay buffer + epsilon schedule
        # -----------------------------
        self.replay = ReplayBuffer(capacity=40000)

        self.epsilon = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 40000
        self.epsilon_step = 0

        # -----------------------------
        # Game counter (for augmentation phases)
        # -----------------------------
        self.game_counter = 0


    # ---------------------------------------------------------
    # Store transition + symmetry augmentation
    # ---------------------------------------------------------
    def store_transition(self, t):
        """
        Store a transition in the replay buffer, with optional symmetry
        augmentation depending on training phase.
        """
        # Always store original
        self.replay.push(t)

        # Phase 1: rotations only
        if 1000 <= self.game_counter < 2000:
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))

        # Phase 2: rotations + flip UD
        if 2000 <= self.game_counter < 3000:
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_ud_transition(t, self.board_size))

        # Phase 3: rotations + flip LR
        if 3000 <= self.game_counter < 4000:
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_lr_transition(t, self.board_size))

        # Phase 4: rotations + both flips
        if 4000 <= self.game_counter < 5000:
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_ud_transition(t, self.board_size))
            self.replay.push(symmetry_utils.flip_lr_transition(t, self.board_size))

        # Phase 5: all symmetries
        if self.game_counter >= 5000:
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_ud_transition(t, self.board_size))
            self.replay.push(symmetry_utils.flip_lr_transition(t, self.board_size))
            self.replay.push(symmetry_utils.flip_ud_lr_transition(t, self.board_size))


    # ---------------------------------------------------------
    # Encode board state for NN input
    # ---------------------------------------------------------
    def encode_state(self, board, current_player):
        """
        Convert board to tensor with perspective of current_player.

        Output shape:
            [1, 2, board_size, board_size]
        """
        if board is None:
            arr = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        else:
            arr = np.array(board, dtype=np.int8, copy=False)

        black = (arr == BLACK).astype(np.float32)
        white = (arr == WHITE).astype(np.float32)

        if current_player == BLACK:
            x = np.stack([black, white], axis=0)
        else:
            x = np.stack([white, black], axis=0)

        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)


    # ---------------------------------------------------------
    # Epsilon decay
    # ---------------------------------------------------------
    def _decay_epsilon(self):
        """Linear epsilon decay."""
        if self.epsilon_step < self.epsilon_decay_steps:
            delta = (self.epsilon - self.epsilon_end) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_end, self.epsilon - delta)
            self.epsilon_step += 1


    # ---------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ---------------------------------------------------------
    def select_action(self, env: Gomoku, legal_moves: list[int]) -> int:
        """
        Epsilon-greedy action selection using policy logits.

        For training:
            - high epsilon initially
            - decays over time

        For evaluation:
            - set epsilon low or zero
        """
        self._decay_epsilon()

        # Random move
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        # NN move
        state_t = self.encode_state(env.board, env.current_player)
        logits = self.policy(state_t).squeeze(0)  # shape [225]

        # Mask illegal moves
        mask = torch.full_like(logits, float("-inf"))
        mask[legal_moves] = 0.0
        masked_logits = logits + mask

        # Softmax sampling
        probs = torch.softmax(masked_logits / max(self.temperature, 1e-6), dim=0)
        action_idx = torch.multinomial(probs, 1).item()
        return action_idx
    
    # ---------------------------------------------------------
    # Standard RL-style training (not AlphaZero)
    # ---------------------------------------------------------
    def train_after_game(
        self,
        batch_size: int = 256,
        policy_coeff: float = 1.0,
        value_coeff: float = 1.0
    ) -> dict:
        """
        Perform backprop after each completed game using replay samples.

        This is a standard actor-critic style update:
            - Policy loss uses advantage * log π(a|s)
            - Value loss uses 1-step TD target
        """

        if len(self.replay) < 10:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        batch = self.replay.sample(batch_size)

        # -----------------------------
        # Build tensors
        # -----------------------------
        states_t      = []
        next_states_t = []
        actions       = []
        rewards       = []
        dones         = []

        for tr in batch:
            states_t.append(self.encode_state(tr.state, tr.player))
            next_states_t.append(self.encode_state(tr.next_state, tr.player))
            actions.append(tr.action)
            rewards.append(tr.reward)
            dones.append(tr.done)

        states_t      = torch.cat(states_t, dim=0)  # [B, 2, H, W]
        next_states_t = torch.cat(next_states_t, 0) # [B, 2, H, W]
        actions_t     = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # -----------------------------
        # Policy loss
        # -----------------------------
        logits = self.policy(states_t)  # [B, 225]
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, actions_t.view(-1, 1)).squeeze(1)

        # -----------------------------
        # Value loss (1-step TD)
        # -----------------------------
        with torch.no_grad():
            next_values = self.value(next_states_t).squeeze(1)
            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_values

        values = self.value(states_t).squeeze(1)
        advantages = (targets - values).detach()

        policy_loss = -(advantages * chosen_log_probs).mean()
        value_loss = nn.MSELoss()(values, targets)

        total_loss = policy_coeff * policy_loss + value_coeff * value_loss

        # -----------------------------
        # Optimize
        # -----------------------------
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        total_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            max_norm=1.0
        )

        self.policy_opt.step()
        self.value_opt.step()

        stats = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total_loss.item()),
        }

        print(f"[Train] Game {self.game_counter} stats: {stats}")
        return stats


    # ---------------------------------------------------------
    # AlphaZero-style training (π from MCTS, z from final result)
    # ---------------------------------------------------------
    def train_after_game_az(
        self,
        batch_size: int = 256,
        policy_coeff: float = 1.0,
        value_coeff: float = 1.0
    ) -> dict:
        """
        AlphaZero-style training:
            - Policy head imitates MCTS visit distribution π
            - Value head predicts final outcome z
        """

        if len(self.replay) < 10:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        # Filter transitions that actually have π and z
        batch = [
            tr for tr in self.replay.sample(batch_size)
            if tr.pi is not None and tr.z is not None
        ]

        if len(batch) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        # -----------------------------
        # Build tensors
        # -----------------------------
        states_t = torch.cat(
            [self.encode_state(tr.state, tr.player) for tr in batch],
            dim=0
        )

        logits = self.policy(states_t)  # [B, 225]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Policy target π
        # pi_t = torch.tensor(
        #     [tr.pi for tr in batch],
        #     dtype=torch.float32,
        #     device=self.device
        # )
        pis = np.stack([tr.pi for tr in batch], axis=0)  # [B, A]
        pi_t = torch.from_numpy(pis).to(self.device, dtype=torch.float32)        

        policy_loss = -(pi_t * log_probs).sum(dim=1).mean()

        # Value target z
        z_t = torch.tensor(
            [tr.z for tr in batch],
            dtype=torch.float32,
            device=self.device
        )

        values = self.value(states_t).squeeze(1)
        value_loss = nn.MSELoss()(values, z_t)

        total_loss = policy_coeff * policy_loss + value_coeff * value_loss

        # -----------------------------
        # Optimize
        # -----------------------------
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        total_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            max_norm=1.0
        )

        self.policy_opt.step()
        self.value_opt.step()

        stats = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total_loss.item()),
        }

        print(f"[TrainAZ] Game {self.game_counter} stats: {stats}")
        return stats

    # ---------------------------------------------------------
    # Reward shaping (post‑game)
    # ---------------------------------------------------------
    @staticmethod
    def compute_rewards(result: GameResult, moves: list[Transition]) -> None:
        """
        Assign rewards to transitions after the game ends.

        AlphaZero normally does NOT use reward shaping — it uses z only.
        But if you want simple RL-style shaping:

            - Winner gets +1
            - Loser gets 0 (or -1 if you want stronger signal)
            - Draw gives 0.5 to everyone

        This function modifies transitions in-place.
        """

        if result == GameResult.DRAW:
            for t in moves:
                t.reward = 0.5
            return

        winner = BLACK if result == GameResult.BLACK_WIN else WHITE

        for t in moves:
            # If you want offence learning, set loser reward to -1.0
            # If you want stability, keep it at 0.0
            t.reward = 1.0 if t.player == winner else 0.0

def gomoku_centrality_bias_for_policy(board_size=15, scale=0.1):
    """
    Compute a centrality heatmap: number of 5‑in‑a‑row lines each cell participates in.
    Used to bias the policy network's final layer toward the center early in training.

    Returns:
        torch.tensor of shape [board_size * board_size]
    """
    heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    win_len = 5

    for r in range(board_size):
        for c in range(board_size):
            count = 0

            # Horizontal
            for cc in range(c - win_len + 1, c + 1):
                if 0 <= cc and cc + win_len - 1 < board_size:
                    count += 1

            # Vertical
            for rr in range(r - win_len + 1, r + 1):
                if 0 <= rr and rr + win_len - 1 < board_size:
                    count += 1

            # Diagonal \
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c + d
                if (
                    0 <= rr and rr + win_len - 1 < board_size
                    and 0 <= cc and cc + win_len - 1 < board_size
                ):
                    count += 1

            # Diagonal /
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c - d
                if (
                    0 <= rr and rr + win_len - 1 < board_size
                    and 0 <= cc - (win_len - 1) and cc < board_size
                ):
                    count += 1

            # Cubic weighting exaggerates center preference
            heatmap[r, c] = count ** 3

    bias = heatmap.flatten()
    bias = bias / bias.max() * scale
    return torch.tensor(bias, dtype=torch.float32)


def gomoku_centrality_bias_for_value(board_size=15, scale=0.1):
    """
    Same as policy bias, but without cubic exaggeration.
    Value net should be more stable and less biased.
    """
    heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    win_len = 5

    for r in range(board_size):
        for c in range(board_size):
            count = 0

            # Horizontal
            for cc in range(c - win_len + 1, c + 1):
                if 0 <= cc and cc + win_len - 1 < board_size:
                    count += 1

            # Vertical
            for rr in range(r - win_len + 1, r + 1):
                if 0 <= rr and rr + win_len - 1 < board_size:
                    count += 1

            # Diagonal \
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c + d
                if (
                    0 <= rr and rr + win_len - 1 < board_size
                    and 0 <= cc and cc + win_len - 1 < board_size
                ):
                    count += 1

            # Diagonal /
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c - d
                if (
                    0 <= rr and rr + win_len - 1 < board_size
                    and 0 <= cc - (win_len - 1) and cc < board_size
                ):
                    count += 1

            heatmap[r, c] = count

    bias = heatmap.flatten()
    bias = bias / bias.max() * scale
    return torch.tensor(bias, dtype=torch.float32)