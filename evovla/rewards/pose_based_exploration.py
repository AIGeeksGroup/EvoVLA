"""
Pose-based Object Exploration (POE) Module

Grounds curiosity in SE(3) pose space to avoid spurious pixel-level novelty.
Uses lightweight world models (forward + inverse) to generate intrinsic rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Optional


class PoseBasedExploration(nn.Module):
    """
    Pose-based Object Exploration (POE)
    
    Uses lightweight world models (forward + inverse model) in SE(3) space
    to model object-end-effector relative pose dynamics, producing
    geometry-grounded curiosity rewards.
    
    Paper: Equations 5-6 implementation
    """
    
    def __init__(
        self,
        device: str = "cuda",
        hidden_dim: int = 256,
        eta: float = 1.0
    ):
        """
        Args:
            device: Computation device
            hidden_dim: Hidden layer dimension (paper: 256)
            eta: Curiosity scale (paper: 1.0)
        """
        super().__init__()
        
        self.device = device
        self.eta = eta
        self.hidden_dim = hidden_dim
        
        # Forward model: (z_t, a_t) -> z_{t+1}
        # z_t: 6D relative pose (3D axis-angle + 3D translation)
        # a_t: 7D action (dx, dy, dz, droll, dpitch, dyaw, dgrip)
        self.forward_model = nn.Sequential(
            nn.Linear(6 + 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        ).to(device)
        
        # Inverse model: (z_t, z_{t+1}) -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(6 + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)
        ).to(device)
        
        # Forward loss history (for progress reward)
        self.forward_loss_history = []
        self.max_history_length = 100
        
        print(f"✓ POE initialized (hidden_dim={hidden_dim}, eta={eta})")
    
    @staticmethod
    def encode_pose(ee_pose: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
        """
        Encode SE(3) relative pose to 6D vector.
        
        Args:
            ee_pose: End-effector pose (4x4 transformation matrix)
            obj_pose: Object pose (4x4 transformation matrix)
        
        Returns:
            z: 6D relative pose vector (3D axis-angle + 3D translation)
        """
        # Compute relative transformation: T_rel = T_ee^{-1} @ T_obj
        T_rel = np.linalg.inv(ee_pose) @ obj_pose
        
        # Rotation part -> axis-angle
        rotation = T_rel[:3, :3]
        axis_angle = Rotation.from_matrix(rotation).as_rotvec()
        
        # Translation part
        translation = T_rel[:3, 3]
        
        # Combine to 6D vector
        z = np.concatenate([axis_angle, translation])
        
        return z
    
    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_t1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            z_t: Current pose (batch_size, 6)
            a_t: Action (batch_size, 7)
            z_t1: Next pose (batch_size, 6), only needed during training
        
        Returns:
            z_t1_pred: Predicted next pose (batch_size, 6)
            a_t_pred: Predicted action (batch_size, 7), only returned during training
        """
        # Forward model prediction
        forward_input = torch.cat([z_t, a_t], dim=-1)
        z_t1_pred = self.forward_model(forward_input)
        
        if z_t1 is not None:
            # Inverse model prediction (during training)
            inverse_input = torch.cat([z_t, z_t1], dim=-1)
            a_t_pred = self.inverse_model(inverse_input)
            return z_t1_pred, a_t_pred
        
        return z_t1_pred, None
    
    def compute_rewards(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_t1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute POE intrinsic rewards.
        
        Args:
            z_t: Current pose (batch_size, 6)
            a_t: Action (batch_size, 7)
            z_t1: Actual next pose (batch_size, 6)
        
        Returns:
            r_cur: Curiosity reward (Equation 5)
            r_prog: Progress reward (Equation 6)
        """
        # Forward model prediction
        z_t1_pred = self.forward_model(torch.cat([z_t, a_t], dim=-1))
        
        # Curiosity reward: prediction error (with stop-gradient)
        # r_cur = eta/2 * ||sg(z_t1_pred) - z_t1||^2
        prediction_error = z_t1_pred.detach() - z_t1
        r_cur = (self.eta / 2) * torch.norm(prediction_error, dim=-1) ** 2
        
        # Compute current forward loss for progress reward
        current_loss = F.mse_loss(z_t1_pred, z_t1, reduction='mean').item()
        self.forward_loss_history.append(current_loss)
        
        # Maintain history length
        if len(self.forward_loss_history) > self.max_history_length:
            self.forward_loss_history.pop(0)
        
        # Progress reward: loss decrease = learning progress
        # r_prog = ReLU(L_F(t-1) - L_F(t))
        if len(self.forward_loss_history) >= 2:
            avg_loss_prev = np.mean(self.forward_loss_history[:-1])
            avg_loss_curr = self.forward_loss_history[-1]
            r_prog = max(0, avg_loss_prev - avg_loss_curr)
        else:
            r_prog = 0.0
        
        return r_cur, torch.tensor(r_prog).to(self.device)
    
    def update(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_t1: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Update world model parameters.
        
        Args:
            z_t: Current pose (batch_size, 6)
            a_t: Action (batch_size, 7)
            z_t1: Actual next pose (batch_size, 6)
            optimizer: Optimizer
        
        Returns:
            loss_dict: Dictionary with forward_loss and inverse_loss
        """
        # Forward and inverse model predictions
        z_t1_pred, a_t_pred = self.forward(z_t, a_t, z_t1)
        
        # Forward model loss
        loss_forward = F.mse_loss(z_t1_pred, z_t1)
        
        # Inverse model loss
        loss_inverse = F.mse_loss(a_t_pred, a_t)
        
        # Combined loss (paper: forward_weight=1.0, inverse_weight=0.1)
        loss = loss_forward + 0.1 * loss_inverse
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'forward_loss': loss_forward.item(),
            'inverse_loss': loss_inverse.item(),
            'total_loss': loss.item()
        }
    
    def reset_history(self):
        """Reset forward loss history (call at new task start)."""
        self.forward_loss_history = []


if __name__ == "__main__":
    # Test POE module
    print("Testing POE module...")
    
    poe = PoseBasedExploration(device="cuda", hidden_dim=256, eta=1.0)
    
    # Create test data
    batch_size = 4
    z_t = torch.randn(batch_size, 6).cuda()
    a_t = torch.randn(batch_size, 7).cuda()
    z_t1 = torch.randn(batch_size, 6).cuda()
    
    # Test forward pass
    z_t1_pred, _ = poe.forward(z_t, a_t)
    print(f"✓ Forward prediction shape: {z_t1_pred.shape}")
    
    # Test reward computation
    r_cur, r_prog = poe.compute_rewards(z_t, a_t, z_t1)
    print(f"✓ Curiosity reward shape: {r_cur.shape}, mean: {r_cur.mean():.4f}")
    print(f"✓ Progress reward: {r_prog:.4f}")
    
    # Test update
    optimizer = torch.optim.Adam(poe.parameters(), lr=3e-4)
    loss_dict = poe.update(z_t, a_t, z_t1, optimizer)
    print(f"✓ World model loss: forward={loss_dict['forward_loss']:.4f}, inverse={loss_dict['inverse_loss']:.4f}")
    
    print(f"\n✓ POE module test passed!")

