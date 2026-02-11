import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Residual block
# -----------------------------
class ResidualBlock(nn.Module):
    """
    Standard AlphaZero-style residual block:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> +x -> ReLU
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# -----------------------------
# Policy network
# -----------------------------
class PolicyNet(nn.Module):
    """
    AlphaZero-style policy head.

    Input:
        x: [B, 2, board_size, board_size]
           channel 0 = current player's stones
           channel 1 = opponent stones

    Output:
        logits: [B, board_size * board_size]
    """
    def __init__(self, input_channels=2, board_size=15, num_blocks=7, channels=64):
        super().__init__()
        self.board_size = board_size

        # Shared trunk
        self.conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        return self.head(x)  # logits (softmax applied outside)


# -----------------------------
# Value network
# -----------------------------
class ValueNet(nn.Module):
    """
    AlphaZero-style value head.

    Input:
        x: [B, 2, board_size, board_size]

    Output:
        value: [B, 1] in [-1, 1]
    """
    def __init__(self, input_channels=2, board_size=15, num_blocks=7, channels=64):
        super().__init__()
        self.board_size = board_size

        # Shared trunk
        self.conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Value head
        self.head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        return self.head(x)  # shape [B, 1]