import torch
import numpy as np
from dataclasses import replace
from utils import Transition

def rotate_board(board: np.ndarray, k: int) -> np.ndarray:
    """Rotate board by 90° * k (k in {1,2,3})."""
    return np.rot90(board, k=k, axes=(0, 1))  # rotate the 2D board

def rotate_board_t(board: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(board, k=k, dims=(0, 1))

def rotate_action(action: int, board_size: int, k: int) -> int:
    """Map flat index action under 90° * k rotation."""
    r, c = divmod(action, board_size)
    if k == 1:  # 90°
        r2, c2 = c, board_size - 1 - r
    elif k == 2:  # 180°
        r2, c2 = board_size - 1 - r, board_size - 1 - c
    elif k == 3:  # 270°
        r2, c2 = board_size - 1 - c, r
    else:
        r2, c2 = r, c
    return r2 * board_size + c2

def rotate_transition(t, board_size: int, k: int):
    """Return a rotated copy of a Transition (state, next_state, action)."""
    # assumes t.state and t.next_state are numpy arrays of shape [C,H,W]
    state_rot = rotate_board(t.state, k)
    next_state_rot = rotate_board(t.next_state, k) if t.next_state is not None else None
    action_rot = rotate_action(t.action, board_size, k) if t.action is not None else None
    return Transition(
        state=state_rot,
        action=action_rot,
        reward=t.reward,
        next_state=next_state_rot,
        done=t.done,
        player=t.player,
        q_value=t.q_value,
        pi=(None if getattr(t, "pi", None) is None else t.pi.copy()),
        z=getattr(t, "z", None),
        meta=(dict(getattr(t, "meta", {})))  # shallow copy if present
    )



def flip_ud(board):  # up/down
    return board[:, ::-1, :]

def flip_lr(board):  # left/right
    return board[:, :, ::-1]

def flip_ud_action(action, board_size):
    r, c = divmod(action, board_size)
    r2, c2 = board_size - 1 - r, c
    return r2 * board_size + c2

def flip_lr_action(action, board_size):
    r, c = divmod(action, board_size)
    r2, c2 = r, board_size - 1 - c
    return r2 * board_size + c2

def flip_lr_transition(t, board_size):
    """Return a left/right flipped copy of a Transition (state, next_state, action)."""
    state_flipped = flip_lr(t.state)
    next_state_flipped = flip_lr(t.next_state) if t.next_state is not None else None
    action_flipped = flip_lr_action(t.action, board_size) if t.action is not None else None
    return replace(t, state=state_flipped, next_state=next_state_flipped, action=action_flipped)

def flip_ud_transition(t, board_size):
    """Return an up/down flipped copy of a Transition (state, next_state, action)."""
    state_flipped = flip_ud(t.state)
    next_state_flipped = flip_ud(t.next_state) if t.next_state is not None else None
    action_flipped = flip_ud_action(t.action, board_size) if t.action is not None else None
    return replace(t, state=state_flipped, next_state=next_state_flipped, action=action_flipped)

def flip_ud_lr_transition(t, board_size):
    """Return a copy of Transition flipped both up/down and left/right."""
    t_ud = flip_ud_transition(t, board_size)
    t_ud_lr = flip_lr_transition(t_ud, board_size)
    return t_ud_lr
