import torch
import traceback

from utils import DEVICE
from policy_and_value_nets import PolicyNet, ValueNet

# Registry of available network classes
NETWORK_REGISTRY = {
    "PolicyNet": PolicyNet,
    "ValueNet": ValueNet,
}


class AgentPersistence:
    """Helper for saving/loading DRL agent state."""

    @staticmethod
    def load(agent, filename="gomoku_agent.pth", load_replay=True, board_size=15):
        try:
            print(f"[Persistence] Trying to load: {filename}")
            print(f"  exists={torch.cuda.is_available() or 'N/A'}, device={getattr(agent, 'device', DEVICE)}")
            checkpoint = torch.load(
                filename,
                map_location=getattr(agent, "device", DEVICE),
                weights_only=False,
            )
            print(f"[Persistence] Loaded checkpoint keys: {list(checkpoint.keys())}")
        except Exception as e:
            print(f"[Persistence] Failed to load {filename}: {type(e).__name__} - {e}")
            traceback.print_exc()
            return

        board_size = checkpoint.get("board_size", board_size)
        model_type = checkpoint.get("model_type", {})

        learning_rate = checkpoint.get("learning_rate", getattr(agent, "lr", 1e-3))
        agent.lr = learning_rate

        # Instantiate correct classes if possible
        if model_type:
            policy_cls = NETWORK_REGISTRY.get(model_type.get("policy"))
            value_cls = NETWORK_REGISTRY.get(model_type.get("value"))
            if policy_cls and value_cls:
                device = getattr(agent, "device", DEVICE)
                agent.policy = policy_cls(board_size=board_size).to(device)
                agent.value = value_cls(board_size=board_size).to(device)
                agent.policy_opt = torch.optim.Adam(
                    agent.policy.parameters(), lr=learning_rate, weight_decay=1e-4
                )
                agent.value_opt = torch.optim.Adam(
                    agent.value.parameters(), lr=learning_rate, weight_decay=1e-4
                )
                print(f"[Persistence] Instantiated {model_type['policy']}/{model_type['value']} before loading.")
            else:
                print(f"[Persistence] Warning: Unknown model types {model_type}. Using existing agent networks.")

        # Load weights
        if "policy" in checkpoint:
            agent.policy.load_state_dict(checkpoint["policy"])
        if "value" in checkpoint:
            agent.value.load_state_dict(checkpoint["value"])

        # Load optimizer states
        if "policy_opt" in checkpoint and hasattr(agent, "policy_opt"):
            agent.policy_opt.load_state_dict(checkpoint["policy_opt"])
        if "value_opt" in checkpoint and hasattr(agent, "value_opt"):
            agent.value_opt.load_state_dict(checkpoint["value_opt"])

        # Metadata
        agent.epsilon = checkpoint.get("epsilon", getattr(agent, "epsilon", 1.0))
        agent.epsilon_step = checkpoint.get("epsilon_step", getattr(agent, "epsilon_step", 0))
        agent.game_counter = checkpoint.get("game_counter", getattr(agent, "game_counter", 0))

        # Replay buffer
        if load_replay and "replay" in checkpoint and hasattr(agent, "replay"):
            agent.replay.buffer = checkpoint["replay"]
            agent.replay.idx = checkpoint.get(
                "replay_idx",
                len(agent.replay.buffer) % agent.replay.capacity,
            )

        if model_type:
            print(
                f"[Persistence] Agent loaded ({model_type.get('policy')}/"
                f"{model_type.get('value')}) from {filename}"
            )
        else:
            print(f"[Persistence] Agent loaded from {filename} (model type not recorded)")

    @staticmethod
    def save(agent, filename="gomoku_agent.pth", save_replay=True):
        checkpoint = {
            "policy": agent.policy.state_dict(),
            "value": agent.value.state_dict(),
            "policy_opt": agent.policy_opt.state_dict(),
            "value_opt": agent.value_opt.state_dict(),
            "learning_rate": getattr(agent, "lr", 1e-3),
            "epsilon": getattr(agent, "epsilon", 1.0),
            "epsilon_step": getattr(agent, "epsilon_step", 0),
            "game_counter": getattr(agent, "game_counter", 0),
            "board_size": getattr(agent.policy, "board_size", 15),
            "model_type": {
                "policy": agent.policy.__class__.__name__,
                "value": agent.value.__class__.__name__,
            },
        }
        if save_replay and hasattr(agent, "replay"):
            checkpoint["replay"] = agent.replay.buffer
            checkpoint["replay_idx"] = agent.replay.idx

        torch.save(checkpoint, filename)
        print(
            f"[Persistence] Agent ({checkpoint['model_type']['policy']}/"
            f"{checkpoint['model_type']['value']}) saved to {filename}"
        )
        if hasattr(agent, "replay"):
            print(
                f"[Persistence] Saved with replay size={len(agent.replay)}, "
                f"epsilon={getattr(agent, 'epsilon', 1.0):.4f}"
            )