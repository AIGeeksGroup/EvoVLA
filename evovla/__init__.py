"""
EvoVLA: Self-Evolving Vision-Language-Action Model

A self-supervised VLA framework that addresses stage hallucination through:
- Stage-Aligned Reward (SAR)
- Pose-Based Object Exploration (POE)  
- Long-Horizon Memory

Paper: "EvoVLA: Self-Evolving Vision-Language-Action Model" (CVPR 2026)
"""

__version__ = "0.1.0"
__author__ = "EvoVLA Team"

from evovla.models.openvla_policy_oft import OpenVLAPolicyOFT
from evovla.rewards.stage_aligned_reward import StageAlignedReward
from evovla.rewards.pose_based_exploration import PoseBasedExploration
from evovla.ppo.ppo_trainer import PPOTrainer

__all__ = [
    "OpenVLAPolicyOFT",
    "StageAlignedReward", 
    "PoseBasedExploration",
    "PPOTrainer",
]

