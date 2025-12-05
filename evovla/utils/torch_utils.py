"""PyTorch utilities for EvoVLA."""

import torch
import numpy as np
import random
from typing import Tuple


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Get computation device."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_tanh_log_prob(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probability for tanh-squashed Gaussian distribution.
    
    Args:
        mu: Mean (batch_size, action_dim)
        log_std: Log standard deviation (batch_size, action_dim)
        action: Action after tanh (batch_size, action_dim)
    
    Returns:
        log_prob: Log probability (batch_size,)
    """
    std = torch.exp(log_std)
    
    # Inverse tanh to get original sample
    # action = tanh(a_tilde), a_tilde = atanh(action)
    eps = 1e-6
    action_clamped = torch.clamp(action, -1 + eps, 1 - eps)
    a_tilde = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))
    
    # Gaussian log probability
    log_prob_gaussian = -0.5 * (
        ((a_tilde - mu) / std) ** 2 +
        2 * log_std +
        np.log(2 * np.pi)
    )
    log_prob_gaussian = log_prob_gaussian.sum(dim=-1)
    
    # Tanh Jacobian correction
    log_det_jacobian = torch.log(1 - action**2 + eps).sum(dim=-1)
    
    return log_prob_gaussian - log_det_jacobian


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.995,
    lambda_: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Reward sequence (T, num_envs)
        values: Value estimates (T+1, num_envs)
        dones: Termination flags (T, num_envs)
        gamma: Discount factor
        lambda_: GAE parameter
    
    Returns:
        advantages: Advantage function (T, num_envs)
        returns: Returns (T, num_envs)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = values[T]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
    
    returns = advantages + values[:-1]
    
    return advantages, returns

